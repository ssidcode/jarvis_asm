"""Speech to text.

Two backends:

* ``faster-whisper`` — local, private, no network. ~0.3-1.5 s on CPU.
* ``groq`` — Groq's hosted Whisper. Typically ~150-350 ms round trip, and it
  skips the ~500 MB local model download entirely. Audio leaves the machine.

Both return the same :class:`Transcript`, so the rest of the pipeline neither
knows nor cares which one ran.
"""
from __future__ import annotations

import io
import logging
import os
import time
import wave
from dataclasses import dataclass

import numpy as np

from .audio_io import to_float32, to_int16

log = logging.getLogger("jarvis.stt")


@dataclass
class Transcript:
    text: str
    language: str = "en"
    duration: float = 0.0
    latency: float = 0.0
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def is_confident(self) -> bool:
        """Whisper hallucinates confident-sounding text on near-silence; these
        two numbers are how you catch it."""
        return self.avg_logprob > -1.0 and self.no_speech_prob < 0.6


class Transcriber:
    def __init__(
        self,
        model: str = "small.en",
        device: str = "auto",
        compute_type: str = "auto",
        language: str | None = "en",
        beam_size: int = 1,
        initial_prompt: str | None = None,
    ):
        self.language = language
        self.beam_size = max(1, int(beam_size))
        self.initial_prompt = initial_prompt
        self._model = self._load(model, device, compute_type)

    @staticmethod
    def _resolve_device(device: str) -> tuple[str, str]:
        if device and device != "auto":
            return device, "float16" if device == "cuda" else "int8"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda", "float16"
        except ImportError:
            pass
        return "cpu", "int8"

    def _load(self, model: str, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        resolved_device, default_ct = self._resolve_device(device)
        ct = default_ct if compute_type in (None, "", "auto") else compute_type
        log.info("loading whisper %s (%s/%s)", model, resolved_device, ct)
        started = time.monotonic()
        instance = WhisperModel(model, device=resolved_device, compute_type=ct)
        log.info("whisper ready in %.1fs", time.monotonic() - started)
        return instance

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        wav = to_float32(np.asarray(audio)).ravel()
        if sample_rate != 16000:
            from .audio_io import resample

            wav = resample(wav, sample_rate, 16000)
        duration = wav.size / 16000
        if duration < 0.1:
            return Transcript("", duration=duration)

        started = time.monotonic()
        segments, info = self._model.transcribe(
            wav,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,  # stops cross-turn hallucination
            initial_prompt=self.initial_prompt,
        )
        parts, logprobs, no_speech = [], [], []
        for seg in segments:
            parts.append(seg.text)
            logprobs.append(getattr(seg, "avg_logprob", 0.0))
            no_speech.append(getattr(seg, "no_speech_prob", 0.0))

        text = " ".join(p.strip() for p in parts).strip()
        result = Transcript(
            text=text,
            language=getattr(info, "language", self.language) or "en",
            duration=duration,
            latency=time.monotonic() - started,
            avg_logprob=float(np.mean(logprobs)) if logprobs else 0.0,
            no_speech_prob=float(np.mean(no_speech)) if no_speech else 0.0,
        )
        log.info(
            "stt (%.2fs audio in %.2fs): %r", duration, result.latency, result.text
        )
        return result


class GroqTranscriber:
    """Groq's hosted Whisper, via the OpenAI-compatible transcriptions route.

    Sends a small in-memory wav rather than writing a temp file; a few seconds
    of 16 kHz mono is only tens of kilobytes, so upload time is negligible.
    """

    #: Groq's production speech models.
    MODELS = ("whisper-large-v3-turbo", "whisper-large-v3")

    def __init__(
        self,
        model: str = "whisper-large-v3-turbo",
        api_key_env: str = "GROQ_API_KEY",
        base_url: str | None = None,
        language: str | None = "en",
        initial_prompt: str | None = None,
        timeout: float = 30.0,
        **_: object,
    ):
        from .llm import GROQ_BASE_URL

        self.model = model
        self.base_url = (base_url or GROQ_BASE_URL).rstrip("/")
        self.language = language
        self.initial_prompt = initial_prompt
        self.timeout = timeout
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise RuntimeError(
                f"{api_key_env} is not set. Get a key at "
                f"https://console.groq.com/keys then:  export {api_key_env}=gsk_..."
            )
        log.info("stt backend: groq %s", self.model)

    @staticmethod
    def _to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(to_int16(np.asarray(audio)).tobytes())
        return buffer.getvalue()

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        import httpx

        pcm = np.asarray(audio).ravel()
        duration = pcm.size / sample_rate
        if duration < 0.1:
            return Transcript("", duration=duration)

        data = {"model": self.model, "response_format": "json"}
        if self.language:
            data["language"] = self.language
        if self.initial_prompt:
            data["prompt"] = self.initial_prompt

        started = time.monotonic()
        try:
            with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=5.0)) as client:
                response = client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": ("speech.wav", self._to_wav_bytes(pcm, sample_rate), "audio/wav")},
                    data=data,
                )
            if response.status_code >= 400:
                log.error("groq stt HTTP %d: %s", response.status_code, response.text[:300])
                return Transcript("", duration=duration)
            text = (response.json().get("text") or "").strip()
        except Exception:
            log.warning("groq stt request failed", exc_info=True)
            return Transcript("", duration=duration)

        result = Transcript(
            text=text,
            language=self.language or "en",
            duration=duration,
            latency=time.monotonic() - started,
            # The API returns no per-segment confidence, so report values that
            # read as confident; empty text is still caught by is_empty.
            avg_logprob=0.0,
            no_speech_prob=0.0,
        )
        log.info("stt (%.2fs audio in %.2fs): %r", duration, result.latency, text)
        return result


def build_stt(cfg_section: dict):
    """Build the configured STT backend, falling back to local Whisper."""
    backend = str(cfg_section.get("backend", "faster-whisper")).lower()
    if backend in ("groq", "groq-whisper"):
        model = cfg_section.get("groq_model") or "whisper-large-v3-turbo"
        try:
            return GroqTranscriber(
                model=model,
                api_key_env=cfg_section.get("api_key_env") or "GROQ_API_KEY",
                base_url=cfg_section.get("base_url"),
                language=cfg_section.get("language", "en"),
                initial_prompt=cfg_section.get("initial_prompt"),
            )
        except Exception as exc:
            log.warning("groq stt unavailable (%s); falling back to local whisper", exc)

    return Transcriber(
        model=cfg_section.get("model", "small.en"),
        device=cfg_section.get("device", "auto"),
        compute_type=cfg_section.get("compute_type", "auto"),
        language=cfg_section.get("language", "en"),
        beam_size=int(cfg_section.get("beam_size", 1) or 1),
        initial_prompt=cfg_section.get("initial_prompt"),
    )
