"""Voice activity detection and utterance endpointing.

Three interchangeable backends:

* ``webrtc``  — webrtcvad. Tiny, fast, no model download. Default.
* ``silero``  — a small neural VAD; better in noise, needs torch + a download.
* ``energy``  — adaptive noise-floor gate. No dependencies, last resort.

On top of whichever backend, :class:`Endpointer` runs the state machine that
decides when you have *stopped talking*: wait for speech to start (with a
timeout), then end the utterance after a run of trailing silence.
"""
from __future__ import annotations

import logging
from collections import deque

import numpy as np

from .audio_io import rms_dbfs, to_float32, to_int16

log = logging.getLogger("jarvis.vad")


# ------------------------------------------------------------- backends
class BaseVAD:
    #: frame lengths (ms) this backend accepts; None = anything
    allowed_frame_ms: tuple[int, ...] | None = None

    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class WebRTCVAD(BaseVAD):
    """webrtcvad only accepts 10/20/30 ms frames, so an 80 ms pipeline frame is
    split into sub-frames and voted on."""

    allowed_frame_ms = (10, 20, 30)

    def __init__(self, aggressiveness: int = 2, sub_frame_ms: int = 20, vote: float = 0.3):
        try:
            import webrtcvad
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "webrtc VAD backend needs `pip install webrtcvad-wheels`"
            ) from exc
        self._vad = webrtcvad.Vad(int(np.clip(aggressiveness, 0, 3)))
        self.sub_frame_ms = sub_frame_ms
        self.vote = vote

    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool:
        pcm = to_int16(np.asarray(frame)).ravel()
        step = int(sample_rate * self.sub_frame_ms / 1000)
        if step <= 0:
            return False
        n_sub = pcm.size // step
        if n_sub == 0:
            return False
        hits = 0
        for i in range(n_sub):
            chunk = pcm[i * step : (i + 1) * step]
            try:
                if self._vad.is_speech(chunk.tobytes(), sample_rate):
                    hits += 1
            except Exception:
                return False
        return (hits / n_sub) >= self.vote


class SileroVAD(BaseVAD):
    """Neural VAD. Wants >=512-sample windows at 16 kHz."""

    def __init__(self, threshold: float = 0.5, window: int = 512):
        import torch

        self.torch = torch
        self.threshold = threshold
        self.window = window
        try:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
        except ImportError:
            self._model, _ = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", trust_repo=True
            )
        self._model.eval()

    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool:
        audio = to_float32(np.asarray(frame)).ravel()
        if audio.size < self.window:
            audio = np.pad(audio, (0, self.window - audio.size))
        best = 0.0
        with self.torch.no_grad():
            for i in range(0, audio.size - self.window + 1, self.window):
                chunk = self.torch.from_numpy(audio[i : i + self.window])
                best = max(best, float(self._model(chunk, sample_rate).item()))
        return best >= self.threshold

    def reset(self) -> None:
        try:
            self._model.reset_states()
        except AttributeError:
            pass


class EnergyVAD(BaseVAD):
    """Adaptive gate using an asymmetric noise-floor tracker.

    The floor must *not* be a plain average of recent levels: during sustained
    speech that average climbs to the speech level, the gate closes, and the
    speaker is cut off mid-sentence. So the floor falls quickly toward quiet
    frames and rises only very slowly, which is the classic minimum-statistics
    behaviour — speech can drag it up a little, silence pulls it straight back.

    ``absolute_floor`` is a hard minimum on the decision threshold, so digital
    silence is never mistaken for speech however low the tracked floor goes.
    """

    def __init__(
        self,
        margin_db: float = 10.0,
        absolute_floor: float = -55.0,
        decay_down: float = 0.5,   # fast adaptation toward quiet
        decay_up: float = 0.01,    # slow adaptation toward loud
        floor_limit: float = -90.0,
        **_: object,
    ):
        self.margin_db = margin_db
        self.absolute_floor = absolute_floor
        self.decay_down = decay_down
        self.decay_up = decay_up
        self.floor_limit = floor_limit
        self._floor = absolute_floor

    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool:
        level = rms_dbfs(np.asarray(frame))
        threshold = max(self._floor + self.margin_db, self.absolute_floor)
        speech = level > threshold
        rate = self.decay_up if level > self._floor else self.decay_down
        self._floor = max(self.floor_limit, self._floor + rate * (level - self._floor))
        return speech

    @property
    def noise_floor(self) -> float:
        return self._floor

    def reset(self) -> None:
        self._floor = self.absolute_floor


def build_vad(backend: str = "webrtc", aggressiveness: int = 2) -> BaseVAD:
    """Make a VAD, degrading gracefully if the preferred backend is missing."""
    backend = (backend or "webrtc").lower()
    order = {
        "webrtc": [("webrtc", WebRTCVAD), ("energy", EnergyVAD)],
        "silero": [("silero", SileroVAD), ("webrtc", WebRTCVAD), ("energy", EnergyVAD)],
        "energy": [("energy", EnergyVAD)],
    }.get(backend, [("webrtc", WebRTCVAD), ("energy", EnergyVAD)])

    for name, cls in order:
        try:
            vad = cls(aggressiveness=aggressiveness) if cls is WebRTCVAD else cls()
            log.info("VAD backend: %s", name)
            return vad
        except Exception as exc:
            log.warning("VAD backend %s unavailable (%s)", name, exc)
    raise RuntimeError("no VAD backend could be constructed")


# ----------------------------------------------------------- endpointing
class Endpointer:
    """Decides when an utterance has ended.

    Feed it frames; it returns a status string each time:

    ``"waiting"``   no speech yet
    ``"speaking"``  speech in progress
    ``"done"``      trailing silence reached — the utterance is complete
    ``"timeout"``   no speech started before ``start_timeout``
    ``"maxlen"``    hit ``max_seconds`` while still talking
    """

    def __init__(
        self,
        vad: BaseVAD,
        sample_rate: int = 16000,
        frame_ms: int = 80,
        silence_ms: int = 800,
        max_seconds: float = 20.0,
        min_seconds: float = 0.4,
        start_timeout: float = 4.0,
        pre_speech_frames: int = 3,
        abandon_silence_multiple: float = 2.5,
    ):
        self.vad = vad
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.silence_frames_needed = max(1, int(silence_ms / frame_ms))
        self.max_frames = int(max_seconds * 1000 / frame_ms)
        self.min_frames = max(1, int(min_seconds * 1000 / frame_ms))
        self.start_timeout_frames = int(start_timeout * 1000 / frame_ms)
        self.pre_speech_frames = pre_speech_frames
        # If a cough or a door slam starts capture but no real speech follows,
        # give up at this much trailing silence instead of waiting out
        # max_seconds with nothing to transcribe.
        self.abandon_silence_frames = max(
            self.silence_frames_needed + 1,
            int(self.silence_frames_needed * abandon_silence_multiple),
        )
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self._frames: list[np.ndarray] = []
        self._pre: deque[np.ndarray] = deque(maxlen=self.pre_speech_frames)
        self._silence_run = 0
        self._speech_frames = 0
        self._total = 0
        self.started = False

    def push(self, frame: np.ndarray) -> str:
        pcm = np.asarray(frame, dtype=np.int16).ravel()
        self._total += 1
        speech = self.vad.is_speech(pcm, self.sample_rate)

        if not self.started:
            # Keep a few frames of pre-roll so the first consonant survives.
            self._pre.append(pcm)
            if speech:
                self.started = True
                self._frames.extend(self._pre)
                self._pre.clear()
                self._speech_frames = 1
                return "speaking"
            if self._total >= self.start_timeout_frames:
                return "timeout"
            return "waiting"

        self._frames.append(pcm)
        if speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self.silence_frames_needed:
                if self._speech_frames >= self.min_frames:
                    return "done"
                if self._silence_run >= self.abandon_silence_frames:
                    # Too little speech to be an utterance; don't sit here
                    # until max_seconds.
                    return "timeout"
        if len(self._frames) >= self.max_frames:
            return "maxlen"
        return "speaking"

    # -- results ------------------------------------------------------
    def audio(self, trim_trailing_silence: bool = True) -> np.ndarray:
        """The captured utterance as int16."""
        if not self._frames:
            return np.zeros(0, dtype=np.int16)
        frames = self._frames
        if trim_trailing_silence and self._silence_run > 1:
            keep = max(1, len(frames) - self._silence_run + 1)
            frames = frames[:keep]
        return np.concatenate(frames)

    @property
    def duration(self) -> float:
        return sum(f.size for f in self._frames) / self.sample_rate

    @property
    def speech_ratio(self) -> float:
        return self._speech_frames / max(1, len(self._frames))
