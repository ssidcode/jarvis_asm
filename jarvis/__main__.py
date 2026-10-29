"""Entry point:  python -m jarvis [--config path] [--check]"""
from __future__ import annotations

import argparse
import sys

from .config import Config
from .logutil import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jarvis", description="Speaker-locked, wake-word voice assistant"
    )
    parser.add_argument("-c", "--config", default=None, help="path to config.yaml")
    parser.add_argument("--check", action="store_true", help="load every stage and exit")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING")
    parser.add_argument(
        "--no-speaker-check",
        action="store_true",
        help="DANGER: disable voice verification for this run",
    )
    args = parser.parse_args(argv)

    try:
        cfg = Config.load(args.config)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.log_level:
        cfg.section("logging")["level"] = args.log_level
    if args.no_speaker_check:
        cfg.section("speaker")["enabled"] = False

    log = setup_logging(cfg.get("logging.level", "INFO"))

    from .assistant import Jarvis

    try:
        jarvis = Jarvis(cfg)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 3
    except Exception as exc:
        log.error("startup failed: %s", exc)
        log.debug("", exc_info=True)
        return 1

    if args.check:
        log.info("all stages loaded successfully — configuration looks good")
        return 0

    jarvis.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
