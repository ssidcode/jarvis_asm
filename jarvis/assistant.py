"""The orchestrator: one state machine over a continuous microphone stream.

    IDLE ──wake word──> VERIFY_WAKE ──voice matches──> LISTEN
     ^                      │ no match                  │
     │                      └──────────> IDLE           │ utterance captured
     │                                                   v
     └── SPEAK <── THINK <── VERIFY_COMMAND <────────────┘
          │                      │ no match
          │ follow-up window     └────────> IDLE
          └──> LISTEN (no wake word needed, still speaker-gated)

Design notes
------------
* One thread owns the microphone and runs every stage, so there is no risk of
  two stages fighting over frames. Heavy work (STT, LLM) blocks the loop, which
  is correct — we genuinely don't want to detect a wake word mid-inference.
* The ring buffer is filled on *every* frame, in every state. That's what makes
  the wake-phrase lookback and the pre-roll possible.
* While speaking, the mic is muted (half-duplex), so Jarvis can't hear itself
  and retrigger. With ``barge_in`` enabled it unmutes and listens for the wake
  word mid-reply instead.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from .audio_io import MicStream, RingBuffer, Speaker, save_wav
from .config import Config, frame_samples
from .llm import Conversation, LLMError, build_llm
from .logutil import TranscriptLog, setup_logging
from .speaker_id import SpeakerVerifier
from .stt import build_stt
from .tts import StreamingSpeaker, build_tts
from .vad import Endpointer, build_vad
from .wakeword import WakeWordDetector

log = logging.getLogger("jarvis.assistant")


class State(Enum):
    IDLE = auto()
    VERIFY_WAKE = auto()
    LISTEN = auto()
    VERIFY_COMMAND = auto()
    THINK = auto()
    SPEAK = auto()


@dataclass
class Stats:
    wake_detections: int = 0
    speaker_rejections: int = 0
    turns_completed: int = 0
    empty_utterances: int = 0
    errors: int = 0
    started: float = field(default_factory=time.monotonic)

    def summary(self) -> str:
        uptime = time.monotonic() - self.started
        return (
            f"up {uptime / 60:.1f} min | wake hits {self.wake_detections} | "
            f"turns {self.turns_completed} | rejected voices {self.speaker_rejections} | "
            f"empty {self.empty_utterances} | errors {self.errors}"
        )


class _BargeInMonitor:
    """Watches the mic while Jarvis is speaking, so you can cut it off.

    Without this, ``barge_in: true`` would be a lie: the speak call blocks the
    main loop, so nobody would be reading frames. This runs a short-lived
    thread that keeps scoring the wake word (and, if verification is on,
    checks the interrupting voice is yours) and flips the interrupt flag.
    """

    def __init__(self, owner: "Jarvis"):
        self.owner = owner
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.triggered = False

    def __enter__(self) -> "_BargeInMonitor":
        self._stop.clear()
        self.triggered = False
        self.owner.mic.unmute()
        self._thread = threading.Thread(target=self._run, name="barge-in", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        owner = self.owner
        while not self._stop.is_set():
            frame = owner.mic.read(timeout=0.2)
            if frame is None:
                continue
            owner.ring.push(frame)
            detection = owner.wake.process(frame)
            if detection is None:
                continue
            # Make sure it is *you* interrupting, not the reply itself leaking
            # back through the speakers or someone else in the room.
            if owner.verifier is not None and owner.cfg.get("speaker.verify_wakeword", True):
                clip = owner.ring.last_seconds(1.3, owner.sample_rate)
                if not owner.verifier.verify_wakeword(clip, owner.sample_rate).accepted:
                    continue
            log.info("barge-in — stopping playback")
            self.triggered = True
            owner.request_interrupt()
            return


class Jarvis:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log = setup_logging(cfg.get("logging.level", "INFO"))
        self.stats = Stats()
        self.state = State.IDLE
        self._running = False

        self.sample_rate = int(cfg.get("audio.sample_rate", 16000))
        self.frame_len = frame_samples(cfg)
        self.frame_ms = int(cfg.get("audio.frame_ms", 80))

        # --- audio plumbing ---
        self.mic = MicStream(
            sample_rate=self.sample_rate,
            frame_samples=self.frame_len,
            device=cfg.get("audio.input_device"),
            gain=float(cfg.get("audio.gain", 1.0) or 1.0),
        )
        self.ring = RingBuffer(
            int(float(cfg.get("audio.lookback_seconds", 3.0)) * self.sample_rate)
        )
        self.speaker_out = Speaker(device=cfg.get("audio.output_device"))

        # --- stages ---
        self.wake = WakeWordDetector(
            model=cfg.get("wakeword.model", "hey_jarvis"),
            threshold=float(cfg.get("wakeword.threshold", 0.5)),
            trigger_level=int(cfg.get("wakeword.trigger_level", 1)),
            refractory_seconds=float(cfg.get("wakeword.refractory_seconds", 1.5)),
            vad_threshold=float(cfg.get("wakeword.vad_threshold", 0.0) or 0.0),
        )
        self.verifier = self._build_verifier()
        self.vad = build_vad(
            cfg.get("vad.backend", "webrtc"), int(cfg.get("vad.aggressiveness", 2))
        )
        self.stt = build_stt(cfg.section("stt"))
        self.llm = build_llm(cfg.section("llm"))
        self.conversation = Conversation(
            cfg.get("llm.system_prompt", "You are a helpful voice assistant."),
            max_turns=int(cfg.get("llm.history_turns", 8)),
        )
        self.voice = StreamingSpeaker(
            build_tts(cfg.section("tts")),
            self.speaker_out,
            enabled=bool(cfg.get("tts.sentence_streaming", True)),
        )
        self.transcript = TranscriptLog(cfg.get("logging.transcript_log"))

        self.stop_phrases = {
            s.lower().strip() for s in cfg.get("conversation.stop_phrases", []) or []
        }
        self._interrupt_requested = False

    # ------------------------------------------------------------ setup
    def _build_verifier(self) -> SpeakerVerifier | None:
        if not self.cfg.get("speaker.enabled", True):
            log.warning(
                "SPEAKER VERIFICATION IS OFF — Jarvis will answer anyone who "
                "says the wake word."
            )
            return None
        return SpeakerVerifier(
            voiceprint_path=self.cfg.path("speaker.voiceprint_path", "~/.jarvis/voiceprint.npz"),
            model=self.cfg.get("speaker.model", "speechbrain/spkrec-ecapa-voxceleb"),
            threshold=float(self.cfg.get("speaker.threshold", 0.45)),
            wakeword_threshold=float(self.cfg.get("speaker.wakeword_threshold", 0.38)),
            min_seconds=float(self.cfg.get("speaker.min_verify_seconds", 0.6)),
        )

    # ------------------------------------------------------- main loop
    def run(self) -> None:
        self._running = True
        log.info("=" * 58)
        log.info("Jarvis listening. Say \"Hey Jarvis\" to start. Ctrl-C to quit.")
        log.info("=" * 58)
        try:
            with self.mic:
                self._loop()
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            self._running = False
            log.info("shutdown — %s", self.stats.summary())

    def _loop(self) -> None:
        while self._running:
            frame = self.mic.read(timeout=1.0)
            if frame is None:
                continue
            self.ring.push(frame)

            detection = self.wake.process(frame)
            if detection is None:
                continue

            self.stats.wake_detections += 1
            try:
                self._handle_turn(detection.score)
            except Exception:
                self.stats.errors += 1
                log.exception("turn failed")
            finally:
                self.wake.reset()
                self.state = State.IDLE
                self.mic.unmute()

    # ------------------------------------------------------------ turn
    def _handle_turn(self, wake_score: float) -> None:
        # ---- stage 1: is it my voice saying the wake word? ----
        self.state = State.VERIFY_WAKE
        if self.verifier is not None and self.cfg.get("speaker.verify_wakeword", True):
            clip = self.ring.last_seconds(1.3, self.sample_rate)
            result = self.verifier.verify_wakeword(clip, self.sample_rate)
            log.info("wake-phrase voice check: %s", result)
            self._maybe_save(clip, "wake_accept" if result.accepted else "wake_reject")
            if not result.accepted:
                self.stats.speaker_rejections += 1
                self.transcript.write(
                    event="rejected_wakeword",
                    similarity=result.similarity,
                    wake_score=wake_score,
                )
                self._reject_feedback()
                return

        self._acknowledge()

        # ---- the conversation: first turn, then the follow-up window ----
        followup_window = float(self.cfg.get("conversation.followup_seconds", 6.0))
        first = True
        while self._running:
            text = self._capture_and_transcribe(first_turn=first)
            if text is None:
                return
            if self._is_stop_phrase(text):
                log.info("stop phrase heard — standing down")
                self.conversation.clear()
                return

            self._respond(text)
            self.stats.turns_completed += 1
            first = False

            if followup_window <= 0:
                return
            log.info("listening for a follow-up for %.1fs (no wake word needed)", followup_window)
            if not self._await_followup(followup_window):
                return

    # ------------------------------------------- capture + verify + stt
    def _capture_and_transcribe(self, first_turn: bool) -> str | None:
        self.state = State.LISTEN
        endpointer = Endpointer(
            self.vad,
            sample_rate=self.sample_rate,
            frame_ms=self.frame_ms,
            silence_ms=int(self.cfg.get("vad.silence_ms", 800)),
            max_seconds=float(self.cfg.get("vad.max_utterance_seconds", 20)),
            min_seconds=float(self.cfg.get("vad.min_utterance_seconds", 0.4)),
            start_timeout=float(self.cfg.get("vad.start_timeout_seconds", 4.0)),
        )
        self.mic.flush()  # drop anything buffered before now

        status = "waiting"
        while self._running:
            frame = self.mic.read(timeout=1.0)
            if frame is None:
                continue
            self.ring.push(frame)
            status = endpointer.push(frame)
            if status in ("done", "timeout", "maxlen"):
                break

        if status == "timeout":
            log.info("no speech after the wake word")
            return None
        if status == "maxlen":
            log.info("hit max utterance length")

        audio = endpointer.audio()
        log.info("captured %.2fs of speech", audio.size / self.sample_rate)

        # ---- stage 2: the real speaker gate, on the full command ----
        if self.verifier is not None and self.cfg.get("speaker.verify_command", True):
            if not first_turn and not self.cfg.get("conversation.followup_requires_speaker", True):
                pass
            else:
                self.state = State.VERIFY_COMMAND
                result = self.verifier.verify(audio, self.sample_rate)
                log.info("command voice check: %s", result)
                self._maybe_save(audio, "cmd_accept" if result.accepted else "cmd_reject")
                if not result.accepted:
                    self.stats.speaker_rejections += 1
                    self.transcript.write(
                        event="rejected_command", similarity=result.similarity
                    )
                    self._reject_feedback()
                    return None

        transcript = self.stt.transcribe(audio, self.sample_rate)
        if transcript.is_empty:
            self.stats.empty_utterances += 1
            log.info("nothing transcribed")
            return None
        if not transcript.is_confident:
            log.info(
                "low-confidence transcript discarded (logprob=%.2f, no_speech=%.2f)",
                transcript.avg_logprob,
                transcript.no_speech_prob,
            )
            self.stats.empty_utterances += 1
            return None
        return transcript.text

    # ------------------------------------------------------- responding
    def _respond(self, text: str) -> None:
        self.state = State.THINK
        log.info("you: %s", text)
        self.conversation.add_user(text)
        system = self.cfg.get("llm.system_prompt", "")
        started = time.monotonic()
        reply = ""

        self._interrupt_requested = False
        barge_in = bool(self.cfg.get("conversation.barge_in", True))
        if not barge_in:
            # Half-duplex: deafen the mic so Jarvis cannot hear its own voice.
            self.mic.mute()

        try:
            self.state = State.SPEAK
            if barge_in:
                with _BargeInMonitor(self) as monitor:
                    reply = self._generate_and_speak(system)
                if monitor.triggered:
                    log.info("reply cut short by barge-in")
            else:
                reply = self._generate_and_speak(system)
        except LLMError as exc:
            self.stats.errors += 1
            log.error("%s", exc)
            self.voice.speak("Sorry, I couldn't reach the language model.")
            self.conversation.add_assistant("")
            return
        finally:
            self.mic.unmute()

        log.info("jarvis: %s", reply)
        self.conversation.add_assistant(reply)
        self.transcript.write(
            event="turn",
            user=text,
            assistant=reply,
            latency=round(time.monotonic() - started, 3),
        )

    def _generate_and_speak(self, system: str) -> str:
        should_stop = lambda: self._interrupt_requested  # noqa: E731
        if self.cfg.get("llm.stream", True):
            stream = self.llm.stream(self.conversation.messages, system)
            return self.voice.speak_stream(stream, should_stop=should_stop)
        reply = self.llm.complete(self.conversation.messages, system)
        self.voice.speak(reply, should_stop=should_stop)
        return reply

    def request_interrupt(self) -> None:
        """Stop the current reply immediately (used by barge-in)."""
        self._interrupt_requested = True
        self.voice.interrupt()

    # ------------------------------------------------------- follow-ups
    def _await_followup(self, seconds: float) -> bool:
        """After a reply, listen briefly for more speech with no wake word.

        Returns True if speech started (caller should capture a new utterance).
        """
        deadline = time.monotonic() + seconds
        self.mic.flush()
        self.vad.reset()
        consecutive_speech = 0
        while self._running and time.monotonic() < deadline:
            frame = self.mic.read(timeout=0.3)
            if frame is None:
                continue
            self.ring.push(frame)
            # A wake word during the window is fine too — just continue.
            self.wake.score(frame)
            if self.vad.is_speech(frame, self.sample_rate):
                consecutive_speech += 1
                if consecutive_speech >= 2:  # ~160 ms of speech
                    log.info("follow-up detected")
                    return True
            else:
                consecutive_speech = 0
        log.info("follow-up window closed")
        self.conversation.clear()
        return False

    # ------------------------------------------------------------ misc
    def _acknowledge(self) -> None:
        try:
            self.speaker_out.beep(880, 110, 0.18)
        except Exception:
            log.debug("ack beep failed", exc_info=True)

    def _reject_feedback(self) -> None:
        mode = str(self.cfg.get("speaker.unknown_speaker_response", "silent")).lower()
        if mode == "beep":
            try:
                self.speaker_out.beep(220, 160, 0.15)
            except Exception:
                pass
        elif mode == "spoken":
            self.voice.speak("Sorry, I don't recognise your voice.")
        # "silent" is the secure default: an unknown speaker learns nothing.

    def _is_stop_phrase(self, text: str) -> bool:
        cleaned = text.lower().strip().strip(".!?,")
        return cleaned in self.stop_phrases

    def _maybe_save(self, audio: np.ndarray, tag: str) -> None:
        if not self.cfg.get("logging.save_audio", False):
            return
        directory = self.cfg.path("logging.audio_dir", "~/.jarvis/logs/audio")
        try:
            save_wav(directory / f"{int(time.time() * 1000)}_{tag}.wav", audio, self.sample_rate)
        except Exception:
            log.debug("audio save failed", exc_info=True)

    def stop(self) -> None:
        self._running = False
        self._interrupt_requested = True
        self.voice.interrupt()
