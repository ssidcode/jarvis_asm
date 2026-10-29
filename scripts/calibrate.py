"""Pick your verification threshold from real data instead of guessing.

    python -m scripts.calibrate                  # record live samples
    python -m scripts.calibrate --impostor-dir ./others

Records a few more clips of *you* (genuine trials) and, if you can get them,
clips of other people or a podcast/TV (impostor trials). It then sweeps the
threshold and reports the false-accept / false-reject rates at each, plus the
Equal Error Rate point, and writes the recommended value into config.yaml.

Without impostor audio it can still tell you your genuine-score distribution,
and recommends a threshold a safe margin below your worst genuine score.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from jarvis.audio_io import MicStream, load_wav
from jarvis.config import Config, frame_samples
from jarvis.logutil import setup_logging
from jarvis.speaker_id import SpeakerEncoder, VoicePrint


def sweep(genuine: np.ndarray, impostor: np.ndarray) -> list[tuple[float, float, float]]:
    """(threshold, false_accept_rate, false_reject_rate) over a grid."""
    rows = []
    for thr in np.arange(0.10, 0.90, 0.01):
        far = float((impostor >= thr).mean()) if impostor.size else 0.0
        frr = float((genuine < thr).mean()) if genuine.size else 0.0
        rows.append((float(thr), far, frr))
    return rows


def equal_error_rate(rows: list[tuple[float, float, float]]) -> tuple[float, float]:
    best = min(rows, key=lambda r: abs(r[1] - r[2]))
    return best[0], (best[1] + best[2]) / 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the speaker threshold")
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("-n", "--num-genuine", type=int, default=6)
    parser.add_argument("--impostor-dir", default=None, help="folder of wavs of OTHER people")
    parser.add_argument("--genuine-dir", default=None, help="skip recording, use these wavs")
    parser.add_argument("--write", action="store_true", help="write the result into config.yaml")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    log = setup_logging(cfg.get("logging.level", "INFO"))
    sample_rate = int(cfg.get("audio.sample_rate", 16000))
    voiceprint = VoicePrint.load(cfg.path("speaker.voiceprint_path", "~/.jarvis/voiceprint.npz"))
    encoder = SpeakerEncoder(cfg.get("speaker.model", "speechbrain/spkrec-ecapa-voxceleb"))
    print(f"voiceprint: {len(voiceprint)} clips, cohesion {voiceprint.cohesion():.3f}\n")

    # ---- genuine trials ----
    genuine_clips: list[np.ndarray] = []
    if args.genuine_dir:
        genuine_clips = [load_wav(p, sample_rate) for p in sorted(Path(args.genuine_dir).expanduser().glob("*.wav"))]
    else:
        from scripts.enroll import grade, record_clip

        prompts = [
            "Hey Jarvis, what time is it?",
            "Hey Jarvis, how's the traffic?",
            "Tell me something interesting.",
            "Hey Jarvis, add milk to the shopping list.",
            "What's on my calendar tomorrow?",
            "Hey Jarvis, thanks, that's all.",
        ]
        print("Recording genuine trials — these must be YOU.\n")
        mic = MicStream(sample_rate, frame_samples(cfg), cfg.get("audio.input_device"))
        with mic:
            i = 0
            while len(genuine_clips) < args.num_genuine:
                print(f"[{len(genuine_clips) + 1}/{args.num_genuine}]  Say: \"{prompts[i % len(prompts)]}\"")
                i += 1
                try:
                    input("          press Enter, then speak... ")
                except (EOFError, KeyboardInterrupt):
                    return 1
                audio = record_clip(mic, sample_rate, int(cfg.get("audio.frame_ms", 80)))
                ok, note = grade(audio, sample_rate)
                print(f"          {'captured' if ok else 'rejected'}: {note}\n")
                if ok:
                    genuine_clips.append(audio)

    if not genuine_clips:
        log.error("no genuine clips collected")
        return 2

    genuine = np.array([voiceprint.score(encoder.embed(c, sample_rate))[0] for c in genuine_clips])

    # ---- impostor trials ----
    impostor = np.array([])
    if args.impostor_dir:
        files = sorted(Path(args.impostor_dir).expanduser().rglob("*.wav"))
        if files:
            print(f"Scoring {len(files)} impostor clip(s)...")
            impostor = np.array([
                voiceprint.score(encoder.embed(load_wav(f, sample_rate), sample_rate))[0]
                for f in files
            ])

    # ---- report ----
    print("\n" + "=" * 62)
    print(f"  genuine   n={genuine.size:<4} mean {genuine.mean():.3f}  min {genuine.min():.3f}")
    if impostor.size:
        print(f"  impostor  n={impostor.size:<4} mean {impostor.mean():.3f}  max {impostor.max():.3f}")
    print("=" * 62)

    if impostor.size:
        rows = sweep(genuine, impostor)
        print("\n  threshold   false-accept   false-reject")
        for thr, far, frr in rows:
            if abs(thr * 100 % 5) < 1e-6:
                print(f"     {thr:.2f}        {far * 100:6.1f}%        {frr * 100:6.1f}%")
        eer_thr, eer = equal_error_rate(rows)
        print(f"\n  Equal Error Rate: {eer * 100:.1f}% at threshold {eer_thr:.2f}")
        # Favour security: pick the lowest threshold with zero false accepts,
        # as long as it doesn't reject you too often.
        zero_far = [r for r in rows if r[1] == 0.0 and r[2] <= 0.10]
        recommended = zero_far[0][0] if zero_far else eer_thr
        rationale = "lowest threshold with no impostor accepted" if zero_far else "equal error rate"
    else:
        margin = 0.10
        recommended = max(0.30, float(genuine.min()) - margin)
        rationale = f"{margin:.2f} below your lowest genuine score (no impostor data)"
        print("\n  No impostor audio supplied, so false-accept rate is unknown.")
        print("  Re-run with --impostor-dir for a security-calibrated number.")

    print(f"\n  RECOMMENDED  speaker.threshold: {recommended:.2f}   ({rationale})")
    print(f"               speaker.wakeword_threshold: {max(0.25, recommended - 0.07):.2f}")

    if args.write:
        path = Path(str(getattr(cfg, "source", "config.yaml")))
        text = path.read_text()
        import re

        text = re.sub(r"(?m)^(  threshold:)\s*[\d.]+", rf"\1 {recommended:.2f}", text, count=1)
        text = re.sub(
            r"(?m)^(  wakeword_threshold:)\s*[\d.]+",
            rf"\1 {max(0.25, recommended - 0.07):.2f}",
            text,
            count=1,
        )
        path.write_text(text)
        print(f"\n  written to {path}")
    else:
        print("\n  (re-run with --write to save these into config.yaml)")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
