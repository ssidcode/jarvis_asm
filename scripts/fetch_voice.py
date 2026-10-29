"""Download a Piper TTS voice.

    python -m scripts.fetch_voice                       # en_US-lessac-medium
    python -m scripts.fetch_voice --voice en_GB-alba-medium
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# voice id -> path fragment on the voices repo
KNOWN = {
    "en_US-lessac-medium": "en/en_US/lessac/medium/en_US-lessac-medium",
    "en_US-amy-medium": "en/en_US/amy/medium/en_US-amy-medium",
    "en_US-ryan-high": "en/en_US/ryan/high/en_US-ryan-high",
    "en_GB-alba-medium": "en/en_GB/alba/medium/en_GB-alba-medium",
    "en_GB-northern_english_male-medium": "en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium",
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {url}\n   -> {dest}")

    def hook(count: int, block: int, total: int) -> None:
        if total > 0:
            pct = min(100, count * block * 100 // total)
            print(f"\r   {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=hook)  # noqa: S310
    print("\r   done")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="en_US-lessac-medium")
    parser.add_argument("--dir", default="~/.jarvis/voices")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        print("\n".join(KNOWN))
        return 0
    if args.voice not in KNOWN:
        print(f"unknown voice {args.voice!r}. Options:\n  " + "\n  ".join(KNOWN), file=sys.stderr)
        return 2

    out_dir = Path(args.dir).expanduser()
    fragment = KNOWN[args.voice]
    model = out_dir / f"{args.voice}.onnx"
    print(f"Downloading {args.voice}...")
    try:
        download(f"{BASE}/{fragment}.onnx", model)
        download(f"{BASE}/{fragment}.onnx.json", out_dir / f"{args.voice}.onnx.json")
    except Exception as exc:
        print(f"\ndownload failed: {exc}", file=sys.stderr)
        print("You can fetch voices by hand from https://huggingface.co/rhasspy/piper-voices")
        return 1
    print(f"\nSet this in config.yaml:\n  tts:\n    piper_voice: {model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
