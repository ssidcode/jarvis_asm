"""Check every stage and report what's broken — run this first when stuck.

    python -m scripts.diagnose

Reports your Python and package versions, which openWakeWord API and model
files are present, whether the voiceprint loads, and whether each configured
backend (STT, LLM, TTS) can actually be reached. Exits non-zero if anything
required is missing.
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label:<28} {detail}")


def version_of(module: str) -> str:
    try:
        import importlib.metadata as md

        return md.version(module)
    except Exception:
        try:
            return getattr(importlib.import_module(module), "__version__", "installed")
        except Exception:
            return "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default=None)
    args = parser.parse_args(argv)
    problems = 0

    print("\n" + "=" * 68)
    print("  JARVIS DIAGNOSTICS")
    print("=" * 68)
    print(f"  python {platform.python_version()}  {platform.system()} {platform.machine()}\n")

    # ---------- packages ----------
    print("-- packages " + "-" * 56)
    required = ["numpy", "yaml", "sounddevice", "openwakeword"]
    optional = ["torch", "speechbrain", "faster_whisper", "httpx", "anthropic",
                "webrtcvad", "soundfile", "pyttsx3", "onnxruntime", "tflite_runtime"]
    # NB: catch Exception, not ImportError — `import sounddevice` raises
    # OSError when the PortAudio *system* library is absent, and a diagnostic
    # tool that crashes on the most common fresh-install problem is useless.
    hints = {
        "sounddevice": "install PortAudio: apt install portaudio19-dev / brew install portaudio",
    }
    for name in required:
        try:
            importlib.import_module(name)
            line(OK, name, version_of(name))
        except Exception as exc:
            line(BAD, name, f"{type(exc).__name__}: {exc}")
            if name in hints:
                print(f"        -> {hints[name]}")
            problems += 1
    for name in optional:
        try:
            importlib.import_module(name)
            line(OK, name, version_of(name))
        except ImportError:
            line(WARN, name, "not installed (optional)")
        except Exception as exc:
            line(WARN, name, f"present but unusable: {type(exc).__name__}")

    # ---------- wake word ----------
    print("\n-- wake word " + "-" * 55)
    try:
        from jarvis.wakeword import WakeWordDetector

        frameworks = WakeWordDetector._available_frameworks()
        line(OK, "inference runtimes", ", ".join(frameworks))
        files = WakeWordDetector._model_files("hey_jarvis", frameworks)
        if files:
            line(OK, "hey_jarvis model file", Path(files[0]).name)
        else:
            line(WARN, "hey_jarvis model file", "absent — will be downloaded on load")
        detector = WakeWordDetector("hey_jarvis", threshold=0.5)
        import numpy as np

        detector.score(np.zeros(1280, dtype=np.int16))
        line(OK, "wake word load", f"label={detector.label!r}")
    except Exception as exc:
        line(BAD, "wake word", f"{type(exc).__name__}: {exc}")
        problems += 1

    # ---------- config + voiceprint ----------
    print("\n-- config & voiceprint " + "-" * 45)
    cfg = None
    try:
        from jarvis.config import Config

        cfg = Config.load(args.config)
        line(OK, "config.yaml", str(getattr(cfg, "source", "?")))
    except Exception as exc:
        line(BAD, "config.yaml", str(exc))
        return problems + 1

    vp_path = cfg.path("speaker.voiceprint_path", "~/.jarvis/voiceprint.npz")
    if vp_path and vp_path.exists():
        try:
            from jarvis.speaker_id import VoicePrint

            vp = VoicePrint.load(vp_path)
            cohesion = vp.cohesion()
            status = OK if cohesion >= 0.6 else WARN
            line(status, "voiceprint", f"{len(vp)} clips, cohesion {cohesion:.3f}")
            if cohesion < 0.6:
                print("        -> clips disagree; re-run `python -m scripts.enroll`")
        except Exception as exc:
            line(BAD, "voiceprint", str(exc))
            problems += 1
    else:
        line(BAD, "voiceprint", f"not found at {vp_path}")
        print("        -> run `python -m scripts.enroll`")
        problems += 1

    # ---------- audio devices ----------
    print("\n-- audio " + "-" * 59)
    try:
        import sounddevice as sd  # may raise OSError without PortAudio

        inputs = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
        if inputs:
            line(OK, "input devices", f"{len(inputs)} found")
            chosen = cfg.get("audio.input_device")
            line(OK, "configured device", str(chosen) if chosen is not None else "system default")
        else:
            line(BAD, "input devices", "none found")
            problems += 1
    except Exception as exc:
        line(BAD, "audio", f"{type(exc).__name__}: {exc}")
        print("        -> install PortAudio (see README)")
        problems += 1

    # ---------- keys ----------
    print("\n-- API keys " + "-" * 56)
    for section in ("llm", "stt"):
        backend = str(cfg.get(f"{section}.backend", ""))
        env_name = cfg.get(f"{section}.api_key_env")
        if backend in ("groq", "groq-whisper"):
            env_name = env_name or "GROQ_API_KEY"
        elif backend == "anthropic":
            env_name = env_name or "ANTHROPIC_API_KEY"
        elif backend in ("faster-whisper", "ollama"):
            line(OK, f"{section} ({backend})", "no key needed")
            continue
        if not env_name:
            continue
        if os.environ.get(env_name):
            value = os.environ[env_name]
            line(OK, f"{section} ({backend})", f"{env_name} set ({value[:6]}...)")
        else:
            line(BAD, f"{section} ({backend})", f"{env_name} is NOT set")
            problems += 1

    # ---------- backends ----------
    print("\n-- backends " + "-" * 56)
    try:
        from jarvis.llm import build_llm

        llm = build_llm(cfg.section("llm"))
        line(OK, "llm", f"{type(llm).__name__} model={getattr(llm, 'model', '?')}")
    except Exception as exc:
        line(BAD, "llm", f"{type(exc).__name__}: {str(exc).splitlines()[0]}")
        problems += 1
    # Deliberately does NOT call build_stt() for the local backend: that
    # constructs a WhisperModel, which downloads ~500 MB on first use. A
    # diagnostic must never trigger a large download as a side effect, so the
    # local path is only reported, not loaded.
    stt_cfg = cfg.section("stt")
    stt_backend = str(stt_cfg.get("backend", "faster-whisper")).lower()
    local_model = stt_cfg.get("model", "small.en")
    if stt_backend in ("groq", "groq-whisper"):
        key_name = stt_cfg.get("api_key_env") or "GROQ_API_KEY"
        if os.environ.get(key_name):
            try:
                from jarvis.stt import GroqTranscriber

                engine = GroqTranscriber(
                    model=stt_cfg.get("groq_model") or "whisper-large-v3-turbo",
                    api_key_env=key_name,
                    base_url=stt_cfg.get("base_url"),
                    language=stt_cfg.get("language", "en"),
                )
                line(OK, "stt", f"GroqTranscriber model={engine.model}")
            except Exception as exc:
                line(BAD, "stt", f"{type(exc).__name__}: {str(exc).splitlines()[0]}")
                problems += 1
        else:
            line(BAD, "stt", f"groq selected but {key_name} is not set")
            print(f"        -> would fall back to local faster-whisper "
                  f"({local_model}), a ~500 MB first-time download")
            problems += 1
    else:
        try:
            importlib.import_module("faster_whisper")
            line(OK, "stt", f"faster-whisper {local_model} (loads on first use)")
        except ImportError:
            line(BAD, "stt", "faster-whisper not installed")
            problems += 1
    try:
        from jarvis.tts import build_tts

        tts = build_tts(cfg.section("tts"))
        name = type(tts).__name__
        line(OK if name != "NullTTS" else WARN, "tts", name)
    except Exception as exc:
        line(BAD, "tts", f"{type(exc).__name__}: {exc}")

    print("\n" + "=" * 68)
    if problems:
        print(f"  {problems} problem(s) found — see the FAIL lines above.")
    else:
        print("  Everything checks out. Run:  python -m jarvis")
    print("=" * 68)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())