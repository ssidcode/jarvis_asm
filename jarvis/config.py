"""Configuration loading: YAML file + environment overrides + dotted access."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATHS = [
    Path.cwd() / "config.yaml",
    Path.home() / ".jarvis" / "config.yaml",
    Path(__file__).resolve().parent.parent / "config.yaml",
]


def _coerce(value: str) -> Any:
    """Turn an env-var string into a bool/int/float/None when it looks like one."""
    low = value.strip().lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"null", "none", ""}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


class Config:
    """Dict-backed config with attribute access and ``a.b.c`` lookups."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    # -- construction -------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        candidates = [Path(path)] if path else DEFAULT_CONFIG_PATHS
        for candidate in candidates:
            if candidate.is_file():
                with candidate.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                cfg = cls(data)
                cfg._apply_env_overrides()
                cfg.source = candidate
                return cfg
        raise FileNotFoundError(
            "No config.yaml found. Looked in: "
            + ", ".join(str(c) for c in candidates)
        )

    def _apply_env_overrides(self) -> None:
        """JARVIS_LLM_MODEL=... overrides llm.model."""
        for env_key, raw in os.environ.items():
            if not env_key.startswith("JARVIS_"):
                continue
            parts = env_key[len("JARVIS_") :].lower().split("_", 1)
            if len(parts) != 2:
                continue
            section, key = parts
            if section in self._data and isinstance(self._data[section], dict):
                self._data[section][key] = _coerce(raw)

    # -- access -------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str, default: Any = None) -> Path | None:
        """Get a config value as an expanded absolute Path."""
        raw = self.get(dotted, default)
        if raw in (None, ""):
            return None
        return Path(str(raw)).expanduser()

    def section(self, name: str) -> dict[str, Any]:
        value = self._data.get(name, {})
        return value if isinstance(value, dict) else {}

    def __getattr__(self, item: str) -> Any:
        data = self.__dict__.get("_data", {})
        if item in data:
            value = data[item]
            return Config(value) if isinstance(value, dict) else value
        raise AttributeError(item)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config({self._data!r})"


def frame_samples(cfg: Config) -> int:
    """Samples per audio frame, derived from sample_rate and frame_ms."""
    return int(cfg.get("audio.sample_rate", 16000) * cfg.get("audio.frame_ms", 80) / 1000)
