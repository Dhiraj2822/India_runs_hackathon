"""
nlp_engine.py — Module 2
Provides all embedding and similarity computation.
The ONLY module that touches the sentence-transformer model.
No scoring logic. No edge cases. Only embeddings and math.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from src.models import CandidateProfile, JobDescription
from config import EMBEDDING_MODEL, DEVICE, LOGS_PATH

# ─── Logging setup ────────────────────────────────────────────────────────────
Path(LOGS_PATH).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOGS_PATH,
    level=logging.WARNING,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

# ─── Module-level singleton — model loaded once per process, never reloaded ───
_model: Optional[SentenceTransformer] = None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_model() -> SentenceTransformer:
    """
    Load BAAI/bge-small-en-v1.5 from local HuggingFace cache.
    Uses module-level singleton — safe to call multiple times, only loads once.
    Raises RuntimeError if model not found (run setup command first).
    """
    global _model
    if _model is not None:
        return _model
    try:
        _model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        _model.eval()
        return _model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{EMBEDDING_MODEL}'. "
            f"Run: python -c \"from sentence_transformers import SentenceTransformer; "
            f"SentenceTransformer('{EMBEDDING_MODEL}')\" to pre-download it.\n"
            f"Original error: {e}"
        )


def compute_embedding(text: str, model: SentenceTransformer) -> np.ndarray:
    """
    Compute a 384-dimensional embedding for one text string.
    Returns zero vector for empty/whitespace-only input.
    Always returns shape (384,) on CPU.
    """
    if not text or not text.strip():
        return np.zeros(384, dtype=np.float32)

    cleaned = text.strip()
    # BGE models work best with instruction prefix for asymmetric tasks
    # For passage encoding (candidates), no prefix needed
    # For query encoding (JD), prefix is: "Represent this sentence for retrieval: "
    # NOTE: this is handled in embed_jd and embed_candidates_batch separately

    embedding = model.encode(
        cleaned,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize for cosine via dot product
        device=DEVICE,
        show_progress_bar=False,
    )
    return embedding.astype(np.float32)


def embed_jd(jd: JobDescription, model: SentenceTransformer) -> JobDescription:
    """
    Compute and store the JD embedding.
    BGE query instruction prefix applied here.
    Call once at startup — never recompute per candidate.
    Modifies jd.embedding in-place and returns jd.
    """
    # BGE-small uses instruction prefix for the query side
    query_text = f"Represent this sentence for retrieval: {jd.raw_text.strip()}"

    jd.embedding = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=DEVICE,
        show_progress_bar=False,
    ).astype(np.float32)

    return jd


def embed_candidates_batch(
    candidates: List[CandidateProfile],
    model: SentenceTransformer,
) -> List[CandidateProfile]:
    """
    Batch embed a list of candidates. Sets candidate.embedding for each.
    Call ONLY on the Stage 1 filtered list (~5,000 candidates).
    DO NOT call on all 100,000 — will exceed time budget.

    Uses batch_size=128 for optimal CPU throughput with BGE-small.
    Passages (candidate text) do NOT use the query prefix.
    """
    texts = [c.full_text for c in candidates]

    embeddings = model.encode(
        texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=DEVICE,
        show_progress_bar=True,
    )

    for candidate, embedding in zip(candidates, embeddings):
        candidate.embedding = embedding.astype(np.float32)

    return candidates


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    Since embeddings are L2-normalised (normalize_embeddings=True above),
    cosine similarity equals dot product — faster computation.
    Returns float in [0.0, 1.0]. Safe for zero vectors.
    """
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    # Dot product of normalised vectors = cosine similarity
    result = float(np.dot(a, b) / (norm_a * norm_b))
    # Clip to [0, 1] — small floating point errors can push slightly outside
    return max(0.0, min(1.0, result))
