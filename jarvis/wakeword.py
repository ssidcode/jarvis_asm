"""Wake word detection — "Hey Jarvis" — via openWakeWord.

openWakeWord ships a pretrained ``hey_jarvis`` model, which is why that exact
phrase costs us nothing to support. It consumes 16 kHz int16 frames (80 ms /
1280 samples is the native stride) and returns a 0-1 score per frame.

The detector adds three things on top of the raw score:

* **trigger_level** — require N consecutive frames over threshold, which kills
  most single-frame false positives from clatter and plosives.
* **refractory period** — after a hit, go deaf briefly so one spoken wake
  phrase cannot fire three times.
* **score history** — kept for the calibration script and DEBUG logging.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("jarvis.wakeword")


@dataclass
class Detection:
    """One accepted wake-word event."""

    label: str
    score: float
    timestamp: float = field(default_factory=time.monotonic)


class WakeWordDetector:
    def __init__(
        self,
        model: str = "hey_jarvis",
        threshold: float = 0.5,
        trigger_level: int = 1,
        refractory_seconds: float = 1.5,
        vad_threshold: float = 0.0,
        inference_framework: str | None = None,
    ):
        self.label = model
        self.threshold = float(threshold)
        self.trigger_level = max(1, int(trigger_level))
        self.refractory_seconds = float(refractory_seconds)
        self._consecutive = 0
        self._last_fire = 0.0
        self._scores: deque[float] = deque(maxlen=256)
        self._model = self._load(model, vad_threshold, inference_framework)

    # -- model loading ------------------------------------------------
    # openWakeWord's API changed across releases, and which release you get
    # depends on your Python version (0.6.x pulls tflite-runtime, which has no
    # wheel for 3.12+, so pip silently resolves 0.4.x there). Both are
    # supported below rather than pinning you to one Python.
    @staticmethod
    def _available_frameworks() -> list[str]:
        """Inference runtimes actually importable here, best first.

        This must drive model-file selection: loading a ``.tflite`` with
        onnxruntime fails with an opaque INVALID_PROTOBUF error, so a model
        format is only ever a candidate if its runtime is present.
        """
        frameworks: list[str] = []
        for module in ("tflite_runtime.interpreter", "tensorflow"):
            try:
                __import__(module)
                frameworks.append("tflite")
                break
            except ImportError:
                continue
        try:
            __import__("onnxruntime")
            frameworks.append("onnx")
        except ImportError:
            pass
        return frameworks or ["onnx"]

    @classmethod
    def _pick_framework(cls) -> str:
        """tflite is faster and lighter; onnx is the portable fallback."""
        return cls._available_frameworks()[0]

    @staticmethod
    def _model_files(name: str, frameworks: list[str]) -> list[str]:
        """Pretrained model files on disk for ``name``, runnable ones first.

        Searches both the package's bundled resources (older versions ship the
        models) and whatever ``download_models`` cached there (newer ones).
        """
        import openwakeword

        suffix_for = {"tflite": ".tflite", "onnx": ".onnx"}
        usable = [suffix_for[f] for f in frameworks if f in suffix_for]

        candidates: list[str] = []
        getter = getattr(openwakeword, "get_pretrained_model_paths", None)
        if callable(getter):
            try:
                candidates.extend(str(p) for p in getter())
            except Exception:
                log.debug("get_pretrained_model_paths failed", exc_info=True)
        resources = Path(openwakeword.__file__).parent / "resources" / "models"
        if resources.is_dir():
            candidates.extend(
                str(p) for p in resources.iterdir() if p.suffix in (".onnx", ".tflite")
            )

        wanted = name.replace("_", "").replace("-", "").lower()
        matches = [
            c for c in dict.fromkeys(candidates)          # de-dupe, keep order
            if wanted in Path(c).stem.replace("_", "").replace("-", "").lower()
            and Path(c).suffix in usable
            and Path(c).is_file()                          # must actually exist
        ]
        matches.sort(key=lambda c: usable.index(Path(c).suffix))
        return matches

    @staticmethod
    def _repair_model_cache(model: str, framework: str) -> list[str]:
        """Fetch any model files missing for ``framework``.

        openWakeWord's own ``download_models()`` decides whether to download
        by checking only the *tflite* copy, so a download interrupted part-way
        leaves a cache with no ``.onnx`` files that never self-heals — and the
        onnx feature models (melspectrogram, embedding) are needed for every
        onnx load, not just the wake word itself. This re-fetches precisely
        what is absent and no-ops once the cache is complete.
        """
        import re

        import openwakeword

        try:
            from openwakeword.utils import download_file
        except ImportError:
            return []

        suffix = ".tflite" if framework == "tflite" else ".onnx"
        target = Path(openwakeword.__file__).parent / "resources" / "models"
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:
            return []

        specs = list(getattr(openwakeword, "FEATURE_MODELS", {}).values())
        entry = getattr(openwakeword, "MODELS", {}).get(model)
        if entry:
            specs.append(entry)

        fetched: list[str] = []
        for spec in specs:
            url = (spec or {}).get("download_url") or ""
            if not url:
                continue
            url = re.sub(r"\.(tflite|onnx)$", suffix, url)
            destination = target / Path(url).name
            if destination.is_file() and destination.stat().st_size > 0:
                continue
            try:
                log.info("fetching missing model file %s", destination.name)
                download_file(url, str(target))
                fetched.append(destination.name)
            except Exception:
                log.debug("could not fetch %s", url, exc_info=True)
        return fetched

    @staticmethod
    def _library_version() -> str:
        """openWakeWord has no __version__ attribute; ask the metadata."""
        import openwakeword

        attr = getattr(openwakeword, "__version__", None)
        if attr:
            return str(attr)
        try:
            import importlib.metadata as md

            return md.version("openwakeword")
        except Exception:
            return "unknown"

    @staticmethod
    def _bundled_model_path(name: str) -> str | None:
        """Find a pretrained model shipped inside the installed package."""
        import openwakeword

        candidates: list[str] = []
        getter = getattr(openwakeword, "get_pretrained_model_paths", None)
        if callable(getter):
            try:
                candidates.extend(str(p) for p in getter())
            except Exception:
                log.debug("get_pretrained_model_paths failed", exc_info=True)
        resources = (
            Path(openwakeword.__file__).parent / "resources" / "models"
        )
        if resources.is_dir():
            candidates.extend(
                str(p) for p in resources.iterdir() if p.suffix in (".onnx", ".tflite")
            )

        wanted = name.replace("_", "").replace("-", "").lower()
        # Prefer tflite when both exist; it is the faster runtime.
        matches = [
            c for c in candidates
            if wanted in Path(c).stem.replace("_", "").replace("-", "").lower()
        ]
        if not matches:
            return None
        matches.sort(key=lambda c: 0 if c.endswith(".tflite") else 1)
        return matches[0]

    def _loaded_labels(self, instance) -> list[str]:
        """Model keys, however this version exposes them.

        ``prediction_buffer`` is empty until the first predict() call on some
        versions, so ``models`` is checked first.
        """
        for attr in ("models", "model_inputs", "prediction_buffer"):
            keys = getattr(instance, attr, None)
            if isinstance(keys, dict) and keys:
                return list(keys.keys())
        return []

    def _load(self, model: str, vad_threshold: float, framework: str | None):
        """Construct the model, trying each known call convention in turn.

        Deliberately does *not* inspect ``Model.__init__``: in openWakeWord
        0.6.x it is decorated, so its signature introspects as ``(*args,
        **kwargs)`` and reveals nothing. Probing by inspection there wrongly
        concludes the API is unrecognised. Trying the calls and catching the
        failures is the only reliable approach across versions.
        """
        import warnings

        import openwakeword
        from openwakeword.model import Model

        version = self._library_version()

        # onnxruntime warns about the absent CUDA provider on every CPU-only
        # machine, which is just noise here.
        warnings.filterwarnings(
            "ignore", message=".*CUDAExecutionProvider.*", category=UserWarning
        )

        # Newer versions download models on demand; older ones bundle them.
        download = getattr(
            getattr(openwakeword, "utils", None), "download_models", None
        )
        if callable(download):
            try:
                download([model])  # idempotent; a no-op once cached
            except Exception:
                log.debug("model download skipped/failed; using cache", exc_info=True)

        frameworks = self._available_frameworks()
        if framework:
            frameworks = [framework] + [f for f in frameworks if f != framework]
        # Make sure the files for our preferred runtime are actually present;
        # a half-downloaded cache is a common and very confusing failure.
        self._repair_model_cache(model, frameworks[0])
        paths = self._model_files(model, frameworks)
        if not paths and len(frameworks) > 1:
            self._repair_model_cache(model, frameworks[1])
            paths = self._model_files(model, frameworks)
        common: dict[str, object] = {}
        if vad_threshold and vad_threshold > 0:
            common["vad_threshold"] = float(vad_threshold)

        # Ordered most- to least-preferred; each is tried until one works.
        attempts: list[tuple[str, dict[str, object]]] = []
        for fw in frameworks:
            attempts.append(
                (f"alias + {fw}",
                 {"wakeword_models": [model], "inference_framework": fw})
            )
        attempts.append(("alias", {"wakeword_models": [model]}))
        for path in paths:
            fw = "tflite" if path.endswith(".tflite") else "onnx"
            attempts += [
                (f"path {Path(path).name} + {fw}",
                 {"wakeword_models": [path], "inference_framework": fw}),
                (f"path {Path(path).name}", {"wakeword_models": [path]}),
                (f"legacy path {Path(path).name}", {"wakeword_model_paths": [path]}),
            ]

        failures: list[str] = []
        instance = None
        for description, kwargs in attempts:
            try:
                instance = Model(**{**kwargs, **common})
            except TypeError as exc:
                # Wrong keyword for this version — expected while probing.
                failures.append(f"{description}: {exc}")
                log.debug("wake word load via %s rejected: %s", description, exc)
                continue
            except Exception as exc:
                failures.append(f"{description}: {type(exc).__name__}: {exc}")
                log.debug("wake word load via %s failed", description, exc_info=True)
                continue
            if self._loaded_labels(instance):
                log.debug("wake word loaded via %s", description)
                break
            failures.append(f"{description}: loaded no models")
            instance = None

        if instance is None:
            detail = "\n  ".join(failures) or "no load strategy was attempted"
            raise RuntimeError(
                f"could not load wake word {model!r} with openwakeword {version}.\n"
                f"  {detail}\n"
                "Run `python -m scripts.diagnose` for a full report."
            )

        available = self._loaded_labels(instance)
        if available and self.label not in available:
            # Keys are the alias on newer versions and the filename stem
            # ("hey_jarvis_v0.1") on older ones.
            wanted = model.replace("_", "").replace("-", "").lower()
            matches = [
                k for k in available
                if wanted in k.replace("_", "").replace("-", "").lower()
            ]
            if not matches:
                raise RuntimeError(
                    f"wake word {model!r} not in loaded models: {available}"
                )
            self.label = matches[0]
        log.info(
            "wake word ready: %r (openwakeword %s, threshold=%.2f, trigger_level=%d)",
            self.label,
            version,
            self.threshold,
            self.trigger_level,
        )
        return instance

    # -- inference ----------------------------------------------------
    def score(self, frame: np.ndarray) -> float:
        """Score one int16 frame. Cheap: ~1 ms of CPU for 80 ms of audio."""
        pcm = np.asarray(frame, dtype=np.int16).ravel()
        predictions = self._model.predict(pcm)
        value = float(predictions.get(self.label, 0.0))
        self._scores.append(value)
        return value

    def process(self, frame: np.ndarray) -> Detection | None:
        """Score a frame and return a Detection when the phrase is accepted."""
        value = self.score(frame)
        now = time.monotonic()

        if now - self._last_fire < self.refractory_seconds:
            self._consecutive = 0
            return None

        if value < self.threshold:
            self._consecutive = 0
            return None

        self._consecutive += 1
        if self._consecutive < self.trigger_level:
            return None

        self._consecutive = 0
        self._last_fire = now
        log.info("wake word detected (score=%.3f)", value)
        return Detection(label=self.label, score=value, timestamp=now)

    # -- housekeeping -------------------------------------------------
    def reset(self) -> None:
        """Clear internal state. Call after a turn so stale model context
        doesn't leak into the next detection."""
        self._consecutive = 0
        try:
            self._model.reset()
        except AttributeError:
            for buf in getattr(self._model, "prediction_buffer", {}).values():
                buf.clear()

    def arm(self) -> None:
        """Drop the refractory lock so the next phrase can fire immediately."""
        self._last_fire = 0.0
        self._consecutive = 0

    @property
    def recent_scores(self) -> list[float]:
        return list(self._scores)
