"""Text to speech, with sentence-level streaming.

Backends, in auto-detect order: Piper (best offline quality, Linux/Pi),
macOS ``say``, then pyttsx3 (cross-platform fallback).

:func:`split_sentences` is what makes replies feel fast: instead of waiting for
the whole LLM response, we speak each sentence the moment it's complete, so
speech starts while the model is still generating.
"""
from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .audio_io import Speaker, load_wav

log = logging.getLogger("jarvis.tts")

# Each lookbehind must include the trailing period: the split point sits just
# *after* it, so "Dr" alone would never match there.
_ABBREV = (
    r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bProf\.)"
    r"(?<!\bSt\.)(?<!\bJr\.)(?<!\bSr\.)(?<!\bvs\.)(?<!\bNo\.)"
    r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\ba\.m\.)(?<!\bp\.m\.)"
)
# Split on the whitespace after a sentence-final mark, keeping any closing
# quote or bracket attached to the sentence it belongs to. Two fixed-width
# lookbehind alternatives, because Python requires a fixed width.
_SENTENCE_SPLIT = re.compile(
    _ABBREV + r'(?:(?<=[.!?])|(?<=[.!?]["\')\]]))\s+(?=\S)'
)


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries without breaking on common abbreviations."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    return parts


def split_buffer(buffer: str) -> tuple[list[str], str]:
    """Split a *streaming* buffer into completed sentences plus the raw tail.

    Unlike :func:`split_sentences` the remainder is returned verbatim —
    whitespace and all — so the next token concatenates cleanly instead of
    producing "Itmay rain".
    """
    cut = 0
    for match in _SENTENCE_SPLIT.finditer(buffer):
        cut = match.end()
    if cut == 0:
        return [], buffer
    return split_sentences(buffer[:cut]), buffer[cut:]


def clean_for_speech(text: str) -> str:
    """Strip markdown and other artifacts that sound terrible read aloud."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"(?<![a-zA-Z0-9])[*_]([^*_]+)[*_](?![a-zA-Z0-9])", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "a link", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{2,}", ". ", text).replace("\n", " ").strip()


# ------------------------------------------------------------- backends
class BaseTTS:
    sample_rate = 22050

    def synthesize(self, text: str) -> np.ndarray | None:
        """Return audio, or None if this backend speaks on its own."""
        raise NotImplementedError

    def speak(self, text: str, speaker: Speaker) -> bool:
        audio = self.synthesize(text)
        if audio is None:
            return True
        return speaker.play(audio, self.sample_rate)


class PiperTTS(BaseTTS):
    def __init__(self, voice_path: str | Path, **_: object):
        self.voice = Path(str(voice_path)).expanduser()
        if not self.voice.exists():
            raise FileNotFoundError(
                f"Piper voice not found: {self.voice}\n"
                "Download one:  python -m scripts.fetch_voice"
            )
        self.binary = shutil.which("piper")
        self._module = None
        if not self.binary:
            try:
                from piper.voice import PiperVoice

                self._module = PiperVoice.load(str(self.voice))
                self.sample_rate = self._module.config.sample_rate
            except ImportError as exc:
                raise FileNotFoundError("piper is not installed") from exc
        else:
            import json

            cfg = self.voice.with_suffix(self.voice.suffix + ".json")
            if cfg.exists():
                try:
                    self.sample_rate = int(
                        json.loads(cfg.read_text())["audio"]["sample_rate"]
                    )
                except (KeyError, ValueError):
                    pass

    def synthesize(self, text: str) -> np.ndarray | None:
        if self._module is not None:
            import io
            import wave

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                self._module.synthesize(text, wf)
            buf.seek(0)
            with wave.open(buf, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            return np.frombuffer(raw, dtype=np.int16)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.wav"
            subprocess.run(
                [self.binary, "--model", str(self.voice), "--output_file", str(out)],
                input=text.encode("utf-8"),
                check=True,
                capture_output=True,
                timeout=60,
            )
            return load_wav(out, self.sample_rate)


class MacSayTTS(BaseTTS):
    def __init__(self, voice: str | None = None, rate: int = 175, **_: object):
        if not shutil.which("say"):
            raise FileNotFoundError("`say` is only available on macOS")
        self.voice = voice
        self.rate = rate

    def synthesize(self, text: str) -> np.ndarray | None:
        cmd = ["say", "-r", str(self.rate)]
        if self.voice:
            cmd += ["-v", self.voice]
        subprocess.run([*cmd, text], check=False, timeout=120)
        return None  # `say` writes to the output device itself


class Pyttsx3TTS(BaseTTS):
    def __init__(self, voice: str | None = None, rate: int = 175, **_: object):
        import pyttsx3

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        if voice:
            self._engine.setProperty("voice", voice)

    def synthesize(self, text: str) -> np.ndarray | None:
        self._engine.say(text)
        self._engine.runAndWait()
        return None


class NullTTS(BaseTTS):
    """Prints instead of speaking — useful on a headless box."""

    def synthesize(self, text: str) -> np.ndarray | None:
        log.info("[tts disabled] %s", text)
        return None


def build_tts(cfg_section: dict) -> BaseTTS:
    backend = str(cfg_section.get("backend", "auto")).lower()
    voice_path = cfg_section.get("piper_voice")
    rate = int(cfg_section.get("rate", 175) or 175)
    voice = cfg_section.get("voice")

    def piper() -> BaseTTS:
        return PiperTTS(voice_path)

    def mac() -> BaseTTS:
        return MacSayTTS(voice, rate)

    def pytts() -> BaseTTS:
        return Pyttsx3TTS(voice, rate)

    if backend == "none":
        return NullTTS()
    explicit = {"piper": [piper], "say": [mac], "pyttsx3": [pytts]}
    if backend in explicit:
        order = explicit[backend]
    elif platform.system() == "Darwin":
        order = [mac, piper, pytts]
    else:
        order = [piper, pytts, mac]

    for factory in order:
        try:
            engine = factory()
            log.info("TTS backend: %s", type(engine).__name__)
            return engine
        except Exception as exc:
            log.debug("TTS backend unavailable: %s", exc)
    log.warning("no TTS backend available — replies will be text only")
    return NullTTS()


# --------------------------------------------------------- streaming out
class StreamingSpeaker:
    """Consumes an LLM token stream and speaks it sentence by sentence."""

    def __init__(self, tts: BaseTTS, speaker: Speaker, enabled: bool = True):
        self.tts = tts
        self.speaker = speaker
        self.enabled = enabled

    def speak_stream(self, chunks: Iterator[str], should_stop=None) -> str:
        """Speak as text arrives. Returns the full text that was produced."""
        if not self.enabled:
            full = "".join(chunks)
            self.speak(full, should_stop)
            return full

        buffer, spoken_full = "", []
        for chunk in chunks:
            if should_stop and should_stop():
                break
            buffer += chunk
            # Only flush *completed* sentences, so prosody stays natural; the
            # unfinished tail stays in the buffer untouched.
            complete, buffer = split_buffer(buffer)
            for sentence in complete:
                spoken_full.append(sentence)
                if not self._say(sentence, should_stop):
                    return " ".join(spoken_full)
        tail = buffer.strip()
        if tail and not (should_stop and should_stop()):
            spoken_full.append(tail)
            self._say(tail, should_stop)
        return " ".join(spoken_full)

    def speak(self, text: str, should_stop=None) -> bool:
        for sentence in split_sentences(text) or ([text] if text.strip() else []):
            if not self._say(sentence, should_stop):
                return False
        return True

    def _say(self, text: str, should_stop=None) -> bool:
        if should_stop and should_stop():
            return False
        cleaned = clean_for_speech(text)
        if not cleaned:
            return True
        try:
            return self.tts.speak(cleaned, self.speaker)
        except Exception:
            log.warning("TTS failed for %r", cleaned[:60], exc_info=True)
            return True

    def interrupt(self) -> None:
        self.speaker.interrupt()
