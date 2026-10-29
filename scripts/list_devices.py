"""Show audio devices, then verify the mic actually works.

    python -m scripts.list_devices
    python -m scripts.list_devices --test        # 5s level meter
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from jarvis.audio_io import MicStream, list_devices, rms_dbfs
from jarvis.config import Config, frame_samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("--test", action="store_true", help="live input level meter")
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args(argv)

    print("\nAUDIO DEVICES\n" + "-" * 62)
    print(list_devices())
    print("-" * 62)
    print("Set audio.input_device in config.yaml to the number you want.\n")

    if not args.test:
        return 0

    cfg = Config.load(args.config)
    mic = MicStream(
        int(cfg.get("audio.sample_rate", 16000)),
        frame_samples(cfg),
        cfg.get("audio.input_device"),
        float(cfg.get("audio.gain", 1.0) or 1.0),
    )
    print(f"Speak for {args.seconds:.0f}s — you want peaks around -20 to -10 dBFS.\n")
    levels = []
    with mic:
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            frame = mic.read(timeout=0.5)
            if frame is None:
                continue
            level = rms_dbfs(frame)
            levels.append(level)
            filled = int(np.clip((level + 60) / 60, 0, 1) * 50)
            print(f"\r  [{'#' * filled}{'.' * (50 - filled)}] {level:6.1f} dBFS", end="", flush=True)
    print("\n")
    if not levels:
        print("No audio captured. Wrong device, or the OS denied mic permission.")
        return 1
    peak = max(levels)
    print(f"peak {peak:.1f} dBFS, median {np.median(levels):.1f} dBFS")
    if peak < -40:
        print("Too quiet — pick another device, move closer, or raise audio.gain.")
    elif peak > -3:
        print("Too hot — lower the input gain to avoid clipping.")
    else:
        print("Levels look good.")
    if mic.dropped_frames:
        print(f"warning: dropped {mic.dropped_frames} frames (CPU contention)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
