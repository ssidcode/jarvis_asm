"""Speaker verification — the gate that makes this assistant respond to you only.

How it works
------------
We do *not* fine-tune a model on your voice. We use a pretrained speaker
encoder (ECAPA-TDNN, trained on VoxCeleb) that maps any utterance to a
192-dimensional embedding where distance means "different speaker". You enroll
once; your clips become a stored voiceprint. At runtime every candidate clip is
embedded and compared by cosine similarity. Above threshold, you; below, ignored.

This is far more robust than fine-tuning on a handful of your own recordings: a
model trained only on you has never seen a negative example and will happily
accept anyone.

Verification happens twice per turn:

1. On the ~1.2 s of audio containing "Hey Jarvis" — a cheap early reject, so a
   stranger's wake word never even starts a recording.
2. On the full command utterance — the real gate, with more audio to judge.

The voiceprint stores every enrollment embedding, not just their mean, and a
candidate is scored against both the centroid and the best-matching individual
clip. That keeps you recognizable across different moods, mic positions and
times of day.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_io import load_wav, to_float32

log = logging.getLogger("jarvis.speaker")

EMBED_SAMPLE_RATE = 16000


# ---------------------------------------------------------------- result
@dataclass
class VerificationResult:
    accepted: bool
    similarity: float
    threshold: float
    reason: str = ""
    centroid_similarity: float = 0.0
    best_clip_similarity: float = 0.0

    def __str__(self) -> str:
        verdict = "ACCEPT" if self.accepted else "REJECT"
        extra = f" ({self.reason})" if self.reason else ""
        return f"{verdict} sim={self.similarity:.3f} thr={self.threshold:.3f}{extra}"


# ------------------------------------------------------------- encoder
class SpeakerEncoder:
    """Wraps SpeechBrain's ECAPA-TDNN encoder. Lazy-loaded and cached."""

    def __init__(self, model: str = "speechbrain/spkrec-ecapa-voxceleb", device: str | None = None):
        self.model_name = model
        self._device = device
        self._classifier = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._classifier is not None:
            return
        import torch

        self._torch = torch
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:  # speechbrain >= 1.0
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # speechbrain 0.5.x
            from speechbrain.pretrained import EncoderClassifier  # type: ignore

        savedir = Path.home() / ".jarvis" / "models" / self.model_name.replace("/", "__")
        savedir.mkdir(parents=True, exist_ok=True)
        log.info("loading speaker encoder %s on %s", self.model_name, self._device)
        self._classifier = EncoderClassifier.from_hparams(
            source=self.model_name,
            savedir=str(savedir),
            run_opts={"device": self._device},
        )

    def embed(self, audio: np.ndarray, sample_rate: int = EMBED_SAMPLE_RATE) -> np.ndarray:
        """L2-normalised embedding of one mono clip."""
        self._ensure_loaded()
        torch = self._torch
        wav = to_float32(np.asarray(audio)).ravel()
        if sample_rate != EMBED_SAMPLE_RATE:
            from .audio_io import resample

            wav = resample(wav, sample_rate, EMBED_SAMPLE_RATE)
        if wav.size < EMBED_SAMPLE_RATE // 4:  # pad clips under 250 ms
            wav = np.pad(wav, (0, EMBED_SAMPLE_RATE // 4 - wav.size))

        tensor = torch.from_numpy(wav).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._classifier.encode_batch(tensor)
        vec = emb.squeeze().detach().cpu().numpy().astype(np.float32)
        return normalize(vec)

    def embed_many(self, clips: list[np.ndarray], sample_rate: int = EMBED_SAMPLE_RATE) -> np.ndarray:
        return np.stack([self.embed(c, sample_rate) for c in clips]) if clips else np.zeros((0, 192), np.float32)


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-9 else vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two embeddings, in [-1, 1]."""
    a, b = normalize(a), normalize(b)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return -1.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


# ----------------------------------------------------------- voiceprint
class VoicePrint:
    """Your enrolled identity: N embeddings plus their centroid."""

    def __init__(self, embeddings: np.ndarray, name: str = "owner", meta: dict | None = None):
        self.embeddings = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        self.name = name
        self.meta = meta or {}
        self.centroid = normalize(self.embeddings.mean(axis=0))

    # -- scoring ------------------------------------------------------
    def score(self, embedding: np.ndarray) -> tuple[float, float, float]:
        """Return (combined, centroid_sim, best_clip_sim).

        The combined score leans on the centroid (stable) but lets a strong
        single-clip match pull it up, which helps when your voice shifts.
        """
        centroid_sim = cosine(embedding, self.centroid)
        per_clip = [cosine(embedding, e) for e in self.embeddings]
        best_clip = max(per_clip) if per_clip else -1.0
        combined = 0.65 * centroid_sim + 0.35 * best_clip
        return combined, centroid_sim, best_clip

    def cohesion(self) -> float:
        """Mean similarity of enrollment clips to the centroid. Below ~0.6 the
        enrollment is inconsistent — re-record in a quieter room."""
        if len(self.embeddings) < 2:
            return 1.0
        return float(np.mean([cosine(e, self.centroid) for e in self.embeddings]))

    # -- persistence --------------------------------------------------
    def save(self, path: str | Path) -> Path:
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            embeddings=self.embeddings,
            centroid=self.centroid,
            name=np.array(self.name),
            meta=np.array(repr(self.meta)),
        )
        try:
            out.chmod(0o600)  # your voiceprint is a credential
        except OSError:
            pass
        log.info("voiceprint saved: %s (%d clips)", out, len(self.embeddings))
        return out

    @classmethod
    def load(cls, path: str | Path) -> "VoicePrint":
        src = Path(path).expanduser()
        if not src.exists():
            raise FileNotFoundError(
                f"No voiceprint at {src}. Run:  python -m scripts.enroll"
            )
        with np.load(src, allow_pickle=False) as data:
            embeddings = data["embeddings"]
            name = str(data["name"]) if "name" in data else "owner"
        return cls(embeddings, name=name)

    def __len__(self) -> int:
        return len(self.embeddings)


# --------------------------------------------------------- verification
class SpeakerVerifier:
    """Runtime gate. Holds the encoder and the loaded voiceprint."""

    def __init__(
        self,
        voiceprint_path: str | Path,
        model: str = "speechbrain/spkrec-ecapa-voxceleb",
        threshold: float = 0.45,
        wakeword_threshold: float = 0.38,
        min_seconds: float = 0.6,
        device: str | None = None,
    ):
        self.encoder = SpeakerEncoder(model, device=device)
        self.voiceprint = VoicePrint.load(voiceprint_path)
        self.threshold = float(threshold)
        self.wakeword_threshold = float(wakeword_threshold)
        self.min_seconds = float(min_seconds)
        cohesion = self.voiceprint.cohesion()
        log.info(
            "voiceprint loaded: %r, %d clips, cohesion=%.3f",
            self.voiceprint.name,
            len(self.voiceprint),
            cohesion,
        )
        if cohesion < 0.6 and len(self.voiceprint) > 1:
            log.warning(
                "enrollment clips disagree (cohesion %.2f). Re-enroll somewhere "
                "quieter for reliable verification.",
                cohesion,
            )

    def verify(
        self,
        audio: np.ndarray,
        sample_rate: int = EMBED_SAMPLE_RATE,
        threshold: float | None = None,
        min_seconds: float | None = None,
    ) -> VerificationResult:
        thr = self.threshold if threshold is None else float(threshold)
        min_s = self.min_seconds if min_seconds is None else float(min_seconds)
        duration = np.asarray(audio).size / sample_rate

        if duration < min_s:
            # Refuse to guess on audio too short to carry speaker identity.
            return VerificationResult(
                False, 0.0, thr, reason=f"clip too short ({duration:.2f}s < {min_s:.2f}s)"
            )

        embedding = self.encoder.embed(audio, sample_rate)
        combined, centroid_sim, best_clip = self.voiceprint.score(embedding)
        accepted = combined >= thr
        return VerificationResult(
            accepted=accepted,
            similarity=combined,
            threshold=thr,
            reason="" if accepted else "voice does not match owner",
            centroid_similarity=centroid_sim,
            best_clip_similarity=best_clip,
        )

    def verify_wakeword(self, audio: np.ndarray, sample_rate: int = EMBED_SAMPLE_RATE) -> VerificationResult:
        """Looser gate for the short wake phrase: ~1 s of audio carries less
        identity, so we only use it to reject obvious non-matches."""
        return self.verify(
            audio,
            sample_rate,
            threshold=self.wakeword_threshold,
            min_seconds=min(self.min_seconds, 0.4),
        )


# ------------------------------------------------------------ enrollment
def build_voiceprint_from_files(
    wav_paths: list[str | Path],
    encoder: SpeakerEncoder | None = None,
    name: str = "owner",
    drop_outliers: bool = True,
) -> VoicePrint:
    """Build a voiceprint from wav files, discarding clips that disagree with
    the rest (a cough, a sibling talking over you, a clipped recording)."""
    encoder = encoder or SpeakerEncoder()
    clips = [load_wav(p) for p in wav_paths]
    embeddings = encoder.embed_many(clips)
    if len(embeddings) == 0:
        raise ValueError("no enrollment audio provided")

    if drop_outliers and len(embeddings) >= 4:
        centroid = normalize(embeddings.mean(axis=0))
        sims = np.array([cosine(e, centroid) for e in embeddings])
        cutoff = max(0.45, float(np.median(sims) - 2.5 * (np.std(sims) + 1e-6)))
        keep = sims >= cutoff
        if keep.sum() >= 3 and keep.sum() < len(embeddings):
            dropped = [
                Path(str(p)).name for p, k in zip(wav_paths, keep, strict=False) if not k
            ]
            log.warning("dropped %d inconsistent enrollment clip(s): %s", len(dropped), dropped)
            embeddings = embeddings[keep]

    return VoicePrint(
        embeddings,
        name=name,
        meta={"n_clips": int(len(embeddings)), "source": "enrollment"},
    )
