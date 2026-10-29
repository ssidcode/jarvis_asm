"""Console + file logging, and the JSONL transcript log."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("jarvis")
    if _CONFIGURED:
        return logger
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric)
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            rich_tracebacks=True, show_path=False, markup=False
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    except ImportError:  # rich is optional
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
        )
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True
    return logger


class TranscriptLog:
    """Append-only JSONL record of every turn, for debugging and tuning."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path).expanduser() if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **fields: Any) -> None:
        if not self.path:
            return
        record = {"ts": time.time(), **fields}
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logging.getLogger("jarvis").debug("transcript write failed", exc_info=True)
