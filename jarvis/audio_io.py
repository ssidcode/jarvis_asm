"""Microphone capture, a lookback ring buffer, and speaker playback.

The whole pipeline speaks one dialect of audio: 16 kHz, mono, int16 frames of
``frame_ms`` milliseconds. Everything downstream either wants int16 (wake word,
webrtcvad) or float32 in [-1, 1] (speaker embedding, Whisper), so the helpers
for converting between the two live here too.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("jarvis.audio")

INT16_MAX = 32768.0


# ---------------------------------------------------------------- helpers
def to_float32(pcm: np.ndarray) -> np.ndarray:
    """int16 PCM -> float32 in [-1, 1]."""
    if pcm.dtype == np.float32:
        return pcm
    return (pcm.astype(np.float32) / INT16_MAX).clip(-1.0, 1.0)


def to_int16(pcm: np.ndarray) -> np.ndarray:
    """float32 in [-1, 1] -> int16 PCM."""
    if pcm.dtype == np.int16:
        return pcm
    return (np.clip(pcm, -1.0, 1.0) * (INT16_MAX - 1)).astype(np.int16)


def rms_dbfs(pcm: np.ndarray) -> float:
    """Loudness of a frame in dBFS; -inf for pure silence."""
    f = to_float32(np.asarray(pcm))
    if f.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(f, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-9))


def save_wav(path: str | Path, pcm: np.ndarray, sample_rate: int = 16000) -> Path:
    """Write a mono wav. Uses soundfile if present, else the stdlib."""
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    data = to_int16(np.asarray(pcm))
    try:
        import soundfile as sf

        sf.write(str(out), data, sample_rate, subtype="PCM_16")
    except ImportError:
        import wave

        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(data.tobytes())
    return out


def load_wav(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Read a wav as float32 mono, resampling if needed."""
    src = Path(path).expanduser()
    try:
        import soundfile as sf

        data, sr = sf.read(str(src), dtype="float32", always_2d=True)
        data = data.mean(axis=1)
    except ImportError:
        import wave

        with wave.open(str(src), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        data = to_float32(np.frombuffer(raw, dtype=np.int16))
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
    if sr != sample_rate:
        data = resample(data, sr, sample_rate)
    return data.astype(np.float32)


def resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resample. Fine for speech; we only ever use it on file loads."""
    if src_rate == dst_rate or pcm.size == 0:
        return pcm.astype(np.float32)
    duration = pcm.size / src_rate
    n_out = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, duration, num=pcm.size, endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, pcm).astype(np.float32)


# ------------------------------------------------------------ ring buffer
class RingBuffer:
    """Fixed-size rolling buffer of int16 samples.

    This is what makes wake-word speaker verification possible: by the time
    openWakeWord fires, the phrase "Hey Jarvis" is already in the past. The
    ring buffer lets us reach back and pull those samples out to run the
    voiceprint check on the exact audio that triggered us.
    """

    def __init__(self, capacity_samples: int):
        self.capacity = int(capacity_samples)
        self._buf = np.zeros(self.capacity, dtype=np.int16)
        self._written = 0
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray) -> None:
        data = np.asarray(frame, dtype=np.int16).ravel()
        n = data.size
        if n == 0:
            return
        if n >= self.capacity:
            with self._lock:
                self._buf[:] = data[-self.capacity :]
                self._written += n
            return
        with self._lock:
            start = self._written % self.capacity
            end = start + n
            if end <= self.capacity:
                self._buf[start:end] = data
            else:
                split = self.capacity - start
                self._buf[start:] = data[:split]
                self._buf[: end - self.capacity] = data[split:]
            self._written += n

    def last(self, n_samples: int) -> np.ndarray:
        """Most recent ``n_samples``, oldest first."""
        n = int(min(n_samples, self.capacity, self._written))
        if n <= 0:
            return np.zeros(0, dtype=np.int16)
        with self._lock:
            end = self._written % self.capacity
            start = end - n
            if start >= 0:
                return self._buf[start:end].copy()
            return np.concatenate((self._buf[start:], self._buf[:end]))

    def last_seconds(self, seconds: float, sample_rate: int = 16000) -> np.ndarray:
        return self.last(int(seconds * sample_rate))

    def clear(self) -> None:
        with self._lock:
            self._buf[:] = 0
            self._written = 0


# -------------------------------------------------------------- mic input
class MicStream:
    """Background microphone reader.

    PortAudio calls us from its own high-priority thread, so the callback does
    nothing but copy bytes into a queue — any real work there causes dropouts.
    The main loop pulls frames with :meth:`read`.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_samples: int = 1280,
        device: int | str | None = None,
        gain: float = 1.0,
        max_queued_frames: int = 100,
    ):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device = device
        self.gain = float(gain)
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queued_frames)
        self._stream = None
        self._muted = threading.Event()
        self.dropped_frames = 0

    # -- lifecycle ----------------------------------------------------
    def start(self) -> "MicStream":
        import sounddevice as sd

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                log.debug("input status: %s", status)
            if self._muted.is_set():
                return
            frame = np.frombuffer(bytes(indata), dtype=np.int16).copy()
            if self.gain != 1.0:
                frame = np.clip(
                    frame.astype(np.float32) * self.gain, -32768, 32767
                ).astype(np.int16)
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                self.dropped_frames += 1
                try:  # drop the oldest frame, keep the newest
                    self._q.get_nowait()
                    self._q.put_nowait(frame)
                except (queue.Empty, queue.Full):
                    pass

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            device=self.device,
            dtype="int16",
            channels=1,
            callback=callback,
        )
        self._stream.start()
        log.info(
            "mic open: %d Hz, %d-sample frames (%.0f ms), device=%s",
            self.sample_rate,
            self.frame_samples,
            1000 * self.frame_samples / self.sample_rate,
            self.device if self.device is not None else "default",
        )
        return self

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # pragma: no cover - driver teardown
                log.debug("mic close failed", exc_info=True)
            self._stream = None

    def __enter__(self) -> "MicStream":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- reading ------------------------------------------------------
    def read(self, timeout: float | None = 1.0) -> np.ndarray | None:
        """Next frame, or None if nothing arrived within ``timeout``."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush(self) -> int:
        """Throw away buffered frames. Call before capturing an utterance so
        you don't transcribe audio from before the wake word."""
        n = 0
        while True:
            try:
                self._q.get_nowait()
                n += 1
            except queue.Empty:
                return n

    # -- half-duplex support -----------------------------------------
    def mute(self) -> None:
        """Stop queueing frames — used while Jarvis talks, so it cannot hear
        itself and re-trigger on its own speech."""
        self._muted.set()
        self.flush()

    def unmute(self) -> None:
        self.flush()
        self._muted.clear()

    @property
    def muted(self) -> bool:
        return self._muted.is_set()


# ------------------------------------------------------------- playback
class Speaker:
    """Blocking playback with an interrupt flag, for barge-in."""

    def __init__(self, sample_rate: int = 22050, device: int | str | None = None):
        self.sample_rate = sample_rate
        self.device = device
        self._stop = threading.Event()

    def play(self, pcm: np.ndarray, sample_rate: int | None = None) -> bool:
        """Play float32/int16 mono audio. Returns False if interrupted."""
        import sounddevice as sd

        data = to_float32(np.asarray(pcm)).ravel()
        if data.size == 0:
            return True
        sr = sample_rate or self.sample_rate
        self._stop.clear()
        chunk = max(256, sr // 20)
        try:
            with sd.OutputStream(
                samplerate=sr, channels=1, dtype="float32", device=self.device
            ) as out:
                for i in range(0, data.size, chunk):
                    if self._stop.is_set():
                        return False
                    out.write(data[i : i + chunk])
        except Exception:
            log.warning("playback failed", exc_info=True)
            return False
        return True

    def interrupt(self) -> None:
        self._stop.set()

    def beep(self, freq: float = 880.0, ms: int = 120, volume: float = 0.2) -> None:
        sr = self.sample_rate
        t = np.arange(int(sr * ms / 1000.0), dtype=np.float32) / sr
        envelope = np.minimum(1.0, np.minimum(t * 60.0, (t[-1] - t) * 60.0 + 1e-6))
        self.play(volume * envelope * np.sin(2 * np.pi * freq * t), sr)


def list_devices() -> str:
    """Human-readable dump of audio devices, for the setup step."""
    import sounddevice as sd

    lines = []
    for idx, dev in enumerate(sd.query_devices()):
        kind = []
        if dev["max_input_channels"]:
            kind.append(f"in:{dev['max_input_channels']}")
        if dev["max_output_channels"]:
            kind.append(f"out:{dev['max_output_channels']}")
        lines.append(
            f"[{idx:2d}] {dev['name']}  ({', '.join(kind)}) "
            f"@ {int(dev['default_samplerate'])} Hz"
        )
    return "\n".join(lines)
