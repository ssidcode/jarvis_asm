"""End-to-end test of the orchestrator with every heavy backend stubbed.

No microphone, no models, no network. Synthetic audio is pushed through the
real Jarvis state machine so the control flow that matters is exercised:

  * the owner's voice gets a reply
  * a stranger who says the wake word is ignored at stage 1
  * a stranger who starts talking after the owner's wake word is ignored at stage 2
  * silence after the wake word times out cleanly
  * a stop phrase ends the conversation
  * follow-up turns need no wake word
  * barge-in cuts a reply short

Run:  python -m pytest tests/ -v     (or just: python tests/test_pipeline.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.assistant import Jarvis, State  # noqa: E402
from jarvis.audio_io import to_int16  # noqa: E402
from jarvis.config import Config  # noqa: E402
from jarvis.llm import Conversation  # noqa: E402
from jarvis.speaker_id import VerificationResult  # noqa: E402
from jarvis.tts import StreamingSpeaker  # noqa: E402
from jarvis.vad import EnergyVAD, Endpointer  # noqa: E402

SR, FRAME_MS = 16000, 80
N = SR * FRAME_MS // 1000
RNG = np.random.default_rng(7)

OWNER, STRANGER, SILENCE = "owner", "stranger", "silence"


# --------------------------------------------------------------- fakes
#: Identity is encoded as a DC offset: the owner is positive, a stranger
#: negative. Unlike a sentinel sample value this survives concatenation and
#: cannot be imitated by chance by the random "speech" noise.
DC_OFFSET = 2000


def frame(kind: str) -> np.ndarray:
    """A frame of fake speech, tagged with who is speaking."""
    if kind == SILENCE:
        return np.zeros(N, dtype=np.int16)
    noise = (0.15 * RNG.standard_normal(N) * 32767).astype(np.int32)
    offset = DC_OFFSET if kind == OWNER else -DC_OFFSET
    return np.clip(noise + offset, -32768, 32767).astype(np.int16)


class FakeMic:
    """Replays a scripted list of frames, then yields silence forever."""

    def __init__(self, script: list[np.ndarray]):
        self.script = list(script)
        self.index = 0
        self._muted = False
        self.dropped_frames = 0
        self.mute_calls = 0

    def read(self, timeout: float | None = 1.0):
        if self.index < len(self.script):
            f = self.script[self.index]
            self.index += 1
            return f
        return frame(SILENCE)

    def flush(self) -> int:
        return 0

    def mute(self) -> None:
        self._muted = True
        self.mute_calls += 1

    def unmute(self) -> None:
        self._muted = False

    @property
    def muted(self) -> bool:
        return self._muted

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def start(self):
        return self

    def stop(self):
        return None


class FakeWake:
    """Fires when it sees a frame carrying the owner/stranger marker and the
    test has armed it for that index."""

    def __init__(self, fire_at: set[int]):
        self.fire_at = set(fire_at)
        self.calls = 0
        self.resets = 0

    def process(self, f):
        from jarvis.wakeword import Detection

        self.calls += 1
        if (self.calls - 1) in self.fire_at:
            return Detection(label="hey_jarvis", score=0.91)
        return None

    def score(self, f):
        return 0.0

    def reset(self):
        self.resets += 1

    def arm(self):
        pass


class FakeVerifier:
    """Accepts audio whose identity marker says OWNER."""

    def __init__(self):
        self.threshold = 0.45
        self.wakeword_threshold = 0.38
        self.min_seconds = 0.6
        self.calls: list[tuple[str, bool]] = []

    @staticmethod
    def _is_owner(audio: np.ndarray) -> bool:
        """Whoever contributed most of the energy decides, as a real speaker
        embedding would: the mean DC offset of the clip."""
        a = np.asarray(audio, dtype=np.float64).ravel()
        if a.size == 0:
            return False
        return float(a.mean()) > 0.0

    def _check(self, audio, stage, thr):
        ok = self._is_owner(audio)
        self.calls.append((stage, ok))
        return VerificationResult(ok, 0.8 if ok else 0.05, thr,
                                  "" if ok else "voice does not match owner")

    def verify(self, audio, sample_rate=SR, threshold=None, min_seconds=None):
        return self._check(audio, "command", threshold or self.threshold)

    def verify_wakeword(self, audio, sample_rate=SR):
        return self._check(audio, "wake", self.wakeword_threshold)


class FakeSTT:
    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.calls = 0

    def transcribe(self, audio, sample_rate=SR):
        from jarvis.stt import Transcript

        text = self.texts[self.calls] if self.calls < len(self.texts) else "hello"
        self.calls += 1
        return Transcript(text=text, duration=audio.size / sample_rate,
                          avg_logprob=-0.2, no_speech_prob=0.1)


class FakeLLM:
    def __init__(self, reply="Sure thing. Anything else?"):
        self.reply = reply
        self.prompts: list[list[dict]] = []

    def stream(self, messages, system):
        self.prompts.append(list(messages))
        for word in self.reply.split(" "):
            yield word + " "

    def complete(self, messages, system):
        return "".join(self.stream(messages, system))


class FakeTTS:
    sample_rate = 22050

    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text, speaker):
        self.spoken.append(text)
        return True


class FakeSpeakerOut:
    def __init__(self):
        self.beeps = 0

    def beep(self, *a, **k):
        self.beeps += 1

    def play(self, *a, **k):
        return True

    def interrupt(self):
        pass


# ------------------------------------------------------------- harness
def build(script, fire_at, stt_texts, *, cfg_overrides=None, reply="Okay. Done."):
    cfg = Config.load(Path(__file__).resolve().parent.parent / "config.yaml")
    base = {
        "conversation": {"followup_seconds": 0.0, "barge_in": False,
                         "stop_phrases": ["stop", "that's all"],
                         "followup_requires_speaker": True},
        "logging": {"level": "WARNING", "save_audio": False, "transcript_log": None},
        "llm": {"stream": True, "system_prompt": "sys", "history_turns": 4},
    }
    for section, values in {**base, **(cfg_overrides or {})}.items():
        cfg.section(section).update(values)

    j = Jarvis.__new__(Jarvis)          # bypass __init__: no models, no mic
    j.cfg = cfg
    j.sample_rate, j.frame_len, j.frame_ms = SR, N, FRAME_MS
    j.state = State.IDLE
    j._running = True
    j._interrupt_requested = False
    from jarvis.assistant import Stats
    from jarvis.audio_io import RingBuffer
    from jarvis.logutil import TranscriptLog, setup_logging

    j.log = setup_logging("WARNING")
    j.stats = Stats()
    j.ring = RingBuffer(int(3.0 * SR))
    j.mic = FakeMic(script)
    j.wake = FakeWake(fire_at)
    j.verifier = FakeVerifier()
    j.vad = EnergyVAD()
    j.stt = FakeSTT(stt_texts)
    j.llm = FakeLLM(reply)
    j.conversation = Conversation("sys", max_turns=4)
    j.tts = FakeTTS()
    j.speaker_out = FakeSpeakerOut()
    j.voice = StreamingSpeaker(j.tts, j.speaker_out, enabled=True)
    j.transcript = TranscriptLog(None)
    j.stop_phrases = {"stop", "that's all"}
    return j


def run_once(j: Jarvis) -> None:
    """Drive the main loop until the scripted frames are consumed."""
    limit = len(j.mic.script) + 60
    while j.mic.index < len(j.mic.script) and limit > 0:
        limit -= 1
        f = j.mic.read()
        j.ring.push(f)
        if j.wake.process(f) is not None:
            j.stats.wake_detections += 1
            try:
                j._handle_turn(0.91)
            finally:
                j.wake.reset()
                j.state = State.IDLE
                j.mic.unmute()


# --------------------------------------------------------------- tests
def test_owner_gets_a_reply():
    script = [frame(SILENCE)] * 2 + [frame(OWNER)] * 18 + [frame(SILENCE)] * 14
    j = build(script, fire_at={2}, stt_texts=["what's the weather"])
    run_once(j)
    assert j.stats.turns_completed == 1, j.stats.summary()
    assert j.stats.speaker_rejections == 0
    assert j.tts.spoken, "nothing was spoken"
    assert [s for s, ok in j.verifier.calls] == ["wake", "command"]
    assert all(ok for _, ok in j.verifier.calls)
    assert j.llm.prompts[0][-1]["content"] == "what's the weather"
    print("  owner -> replied:", j.tts.spoken)


def test_stranger_wake_word_rejected_at_stage_one():
    script = [frame(SILENCE)] * 2 + [frame(STRANGER)] * 18 + [frame(SILENCE)] * 10
    j = build(script, fire_at={2}, stt_texts=["delete everything"])
    run_once(j)
    assert j.stats.turns_completed == 0
    assert j.stats.speaker_rejections == 1
    assert j.stt.calls == 0, "stranger audio must never reach the transcriber"
    assert not j.tts.spoken, "must stay silent for an unknown speaker"
    assert [s for s, _ in j.verifier.calls] == ["wake"], "should reject before listening"
    print("  stranger wake word -> ignored, STT never ran")


def test_stranger_command_rejected_at_stage_two():
    # Owner says the wake word, a different person gives the command.
    script = ([frame(SILENCE)] * 2 + [frame(OWNER)] * 4
              + [frame(STRANGER)] * 16 + [frame(SILENCE)] * 14)
    j = build(script, fire_at={2}, stt_texts=["transfer the money"])
    run_once(j)
    assert j.stats.turns_completed == 0
    assert j.stats.speaker_rejections == 1
    assert j.stt.calls == 0
    assert not j.tts.spoken
    stages = [s for s, ok in j.verifier.calls]
    assert stages == ["wake", "command"] and j.verifier.calls[-1][1] is False
    print("  stranger command after owner's wake word -> rejected at stage 2")


def test_silence_after_wake_word_times_out():
    script = [frame(SILENCE)] * 2 + [frame(OWNER)] * 3 + [frame(SILENCE)] * 70
    j = build(script, fire_at={2}, stt_texts=["should not be used"],
              cfg_overrides={"vad": {"start_timeout_seconds": 1.0, "silence_ms": 800,
                                     "backend": "energy", "max_utterance_seconds": 20,
                                     "min_utterance_seconds": 0.4}})
    run_once(j)
    assert j.stats.turns_completed == 0
    assert j.stt.calls == 0
    print("  silence after wake word -> clean timeout, no turn")


def test_stop_phrase_ends_turn():
    script = [frame(SILENCE)] * 2 + [frame(OWNER)] * 18 + [frame(SILENCE)] * 14
    j = build(script, fire_at={2}, stt_texts=["that's all"])
    run_once(j)
    assert j.stats.turns_completed == 0
    assert not j.tts.spoken, "a stop phrase should not produce a reply"
    assert j.conversation.messages == []
    print("  stop phrase -> stood down without replying")


def test_followup_needs_no_wake_word():
    script = ([frame(SILENCE)] * 2 + [frame(OWNER)] * 18 + [frame(SILENCE)] * 12
              + [frame(OWNER)] * 18 + [frame(SILENCE)] * 14)
    j = build(script, fire_at={2}, stt_texts=["first question", "second question"],
              cfg_overrides={"conversation": {"followup_seconds": 2.0, "barge_in": False,
                                              "followup_requires_speaker": True,
                                              "stop_phrases": ["stop"]}})
    run_once(j)
    assert j.stats.turns_completed == 2, j.stats.summary()
    assert j.wake.calls >= 1
    # the second turn carried the first one's history
    assert len(j.llm.prompts[1]) > len(j.llm.prompts[0])
    assert j.llm.prompts[1][-1]["content"] == "second question"
    print("  follow-up -> second turn with history, no second wake word")


def test_barge_in_interrupts_reply():
    j = build([frame(SILENCE)], fire_at=set(), stt_texts=[],
              reply="One. Two. Three. Four. Five.")
    spoken_before_stop = []

    class Interrupting(FakeTTS):
        def speak(self, text, speaker):
            spoken_before_stop.append(text)
            j.request_interrupt()      # user cuts in after the first sentence
            return True

    j.tts = Interrupting()
    j.voice = StreamingSpeaker(j.tts, j.speaker_out, enabled=True)
    j.conversation.add_user("tell me a long story")
    reply = j._generate_and_speak("sys")
    assert len(spoken_before_stop) == 1, spoken_before_stop
    assert reply == "One."
    print("  barge-in -> playback stopped after", spoken_before_stop)


def test_half_duplex_mutes_mic_while_speaking():
    script = [frame(SILENCE)] * 2 + [frame(OWNER)] * 18 + [frame(SILENCE)] * 14
    j = build(script, fire_at={2}, stt_texts=["hello there"],
              cfg_overrides={"conversation": {"followup_seconds": 0.0, "barge_in": False,
                                              "stop_phrases": [], "followup_requires_speaker": True}})
    run_once(j)
    assert j.mic.mute_calls >= 1, "mic must be muted while speaking in half-duplex"
    assert not j.mic.muted, "mic must be unmuted again afterwards"
    print("  half-duplex -> mic muted during playback, restored after")


def test_short_clip_is_refused_not_guessed():
    from jarvis.speaker_id import SpeakerVerifier

    sv = SpeakerVerifier.__new__(SpeakerVerifier)
    sv.threshold, sv.wakeword_threshold, sv.min_seconds = 0.45, 0.38, 0.6
    r = sv.verify(np.zeros(1000, dtype=np.int16), SR)
    assert not r.accepted and "short" in r.reason
    print("  sub-threshold clip ->", r)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    print()
    for test in TESTS:
        name = test.__name__
        try:
            test()
            print(f"PASS  {name}\n")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback

            print(f"FAIL  {name}: {exc}")
            traceback.print_exc()
            print()
    total = len(TESTS)
    print("=" * 58)
    print(f"  {total - failures}/{total} passed")
    print("=" * 58)
    sys.exit(1 if failures else 0)
