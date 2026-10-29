"""Enroll your voice — run this once before using Jarvis.

    python -m scripts.enroll

Records a set of prompted phrases, embeds them with ECAPA-TDNN, drops any clip
that disagrees with the rest, and writes ~/.jarvis/voiceprint.npz.

Advice that actually matters for accuracy:
  * Record where you'll normally use it, with the mic you'll normally use.
  * Vary it: normal voice, quieter, slightly further away, a bit faster.
  * Don't whisper or shout — the voiceprint should cover your usual range.
  * Re-run with --append after a few days to make it more robust.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from jarvis.audio_io import MicStream, rms_dbfs, save_wav
from jarvis.config import Config, frame_samples
from jarvis.logutil import setup_logging
from jarvis.speaker_id import SpeakerEncoder, VoicePrint, cosine, normalize
from jarvis.vad import EnergyVAD, build_vad

PHRASES = [
    "Hey Jarvis",
    "Hey Jarvis, what's the weather like today?",
    "Hey Jarvis, set a timer for ten minutes.",
    "Hey Jarvis, play some music.",
    "The quick brown fox jumps over the lazy dog.",
    "Hey Jarvis, remind me to call back this afternoon.",
    "I'd like to know how long the trip usually takes.",
    "Hey Jarvis, turn off the lights in the living room.",
]


def record_clip(
    mic: MicStream,
    sample_rate: int,
    frame_ms: int,
    max_seconds: float = 6.0,
    silence_ms: int = 900,
) -> np.ndarray:
    """Record until you stop talking."""
    from jarvis.vad import Endpointer

    endpointer = Endpointer(
        build_vad("webrtc"),
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        silence_ms=silence_ms,
        max_seconds=max_seconds,
        min_seconds=0.3,
        start_timeout=8.0,
    )
    mic.flush()
    while True:
        frame = mic.read(timeout=1.0)
        if frame is None:
            continue
        status = endpointer.push(frame)
        if status in ("done", "maxlen"):
            return endpointer.audio()
        if status == "timeout":
            return np.zeros(0, dtype=np.int16)


def grade(audio: np.ndarray, sample_rate: int) -> tuple[bool, str]:
    """Reject clips that would poison the voiceprint."""
    duration = audio.size / sample_rate
    if duration < 0.5:
        return False, f"too short ({duration:.1f}s)"
    level = rms_dbfs(audio)
    if level < -45:
        return False, f"too quiet ({level:.0f} dBFS) — move closer to the mic"
    peak = float(np.max(np.abs(audio.astype(np.float32)))) / 32768.0
    if peak > 0.99:
        return False, "clipping — lower the input gain or back off a little"
    return True, f"{duration:.1f}s at {level:.0f} dBFS"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enroll your voice for Jarvis")
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("-n", "--num-clips", type=int, default=8)
    parser.add_argument("--name", default="owner")
    parser.add_argument("--append", action="store_true", help="add to the existing voiceprint")
    parser.add_argument("--keep-wavs", action="store_true", help="also save the raw recordings")
    parser.add_argument("--from-dir", default=None, help="build from existing wavs instead of recording")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    log = setup_logging(cfg.get("logging.level", "INFO"))
    out_path = cfg.path("speaker.voiceprint_path", "~/.jarvis/voiceprint.npz")
    sample_rate = int(cfg.get("audio.sample_rate", 16000))
    encoder = SpeakerEncoder(cfg.get("speaker.model", "speechbrain/spkrec-ecapa-voxceleb"))

    # --- offline path: build from a folder of wavs ---
    if args.from_dir:
        from jarvis.speaker_id import build_voiceprint_from_files

        wavs = sorted(Path(args.from_dir).expanduser().glob("*.wav"))
        if not wavs:
            log.error("no .wav files in %s", args.from_dir)
            return 2
        print(f"Building voiceprint from {len(wavs)} file(s)...")
        voiceprint = build_voiceprint_from_files(wavs, encoder, name=args.name)
        voiceprint.save(out_path)
        print(f"\nDone. cohesion={voiceprint.cohesion():.3f}  ->  {out_path}")
        return 0

    # --- interactive recording ---
    clips: list[np.ndarray] = []
    print("\n" + "=" * 62)
    print("  VOICE ENROLLMENT")
    print("=" * 62)
    print(f"  You'll read {args.num_clips} short phrases out loud.")
    print("  Speak naturally, at the distance you'll normally use.")
    print("  Recording stops automatically when you stop talking.")
    print("=" * 62 + "\n")

    mic = MicStream(
        sample_rate=sample_rate,
        frame_samples=frame_samples(cfg),
        device=cfg.get("audio.input_device"),
        gain=float(cfg.get("audio.gain", 1.0) or 1.0),
    )
    wav_dir = Path.home() / ".jarvis" / "enrollment"

    with mic:
        index = 0
        while len(clips) < args.num_clips:
            phrase = PHRASES[index % len(PHRASES)]
            index += 1
            print(f"[{len(clips) + 1}/{args.num_clips}]  Say:  \"{phrase}\"")
            try:
                input("          press Enter, then speak... ")
            except (EOFError, KeyboardInterrupt):
                print("\ncancelled")
                return 1
            print("          listening...", end="", flush=True)
            time.sleep(0.15)
            audio = record_clip(mic, sample_rate, int(cfg.get("audio.frame_ms", 80)))
            ok, note = grade(audio, sample_rate)
            if not ok:
                print(f"\r          rejected: {note}. Let's try that one again.\n")
                continue
            clips.append(audio)
            print(f"\r          captured ({note})\n")
            if args.keep_wavs:
                save_wav(wav_dir / f"clip_{len(clips):02d}.wav", audio, sample_rate)

    print("Embedding clips...")
    embeddings = encoder.embed_many(clips, sample_rate)

    # Report pairwise agreement so a bad enrollment is visible immediately.
    centroid = normalize(embeddings.mean(axis=0))
    sims = np.array([cosine(e, centroid) for e in embeddings])
    print("\n  clip  similarity to your centroid")
    for i, sim in enumerate(sims, 1):
        bar = "#" * int(max(0, sim) * 40)
        print(f"  {i:>4}  {sim:.3f}  {bar}")

    keep = sims >= max(0.45, float(np.median(sims) - 2.5 * (np.std(sims) + 1e-6)))
    if keep.sum() >= 3 and keep.sum() < len(embeddings):
        print(f"\n  dropping {int((~keep).sum())} inconsistent clip(s)")
        embeddings = embeddings[keep]

    if args.append and Path(str(out_path)).exists():
        existing = VoicePrint.load(out_path)
        embeddings = np.vstack([existing.embeddings, embeddings])
        print(f"  appending to {len(existing)} existing clip(s)")

    voiceprint = VoicePrint(embeddings, name=args.name, meta={"n_clips": int(len(embeddings))})
    voiceprint.save(out_path)
    cohesion = voiceprint.cohesion()

    print("\n" + "=" * 62)
    print(f"  Voiceprint saved: {out_path}")
    print(f"  clips: {len(voiceprint)}   cohesion: {cohesion:.3f}")
    if cohesion < 0.6:
        print("  WARNING: low cohesion. Re-run somewhere quieter for better accuracy.")
    else:
        print("  Quality looks good.")
    print("\n  Next:  python -m scripts.calibrate     (pick your threshold)")
    print("  Then:  python -m jarvis")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
