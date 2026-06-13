# MODULE 2 — NLP ENGINE
## File to build: `src/nlp_engine.py`
## Depends on: `src/models.py`, `config.py`
## Do NOT start this module until ALL 12 Module 1 tests pass.

---

## WHAT THIS MODULE DOES

Provides all embedding and similarity computation.
This is the only module that touches the sentence-transformer model.
No scoring logic lives here. No edge cases. Only embeddings and math.

---

## THE MODEL CHOICE — WHY BAAI/bge-small-en-v1.5

- Size: 133 MB — fits comfortably in 16 GB RAM
- Optimised specifically for retrieval/ranking tasks (what we're doing)
- Better recall than all-MiniLM-L6-v2 on information retrieval benchmarks
- CPU inference: ~18ms per candidate on a modern CPU
- Pre-downloaded during setup — no network access during ranking

Do NOT change the model without updating the timing estimates in TRD.md and re-testing.

---

## IMPORTS

```python
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from src.models import CandidateProfile, JobDescription
from config import EMBEDDING_MODEL, DEVICE, LOGS_PATH
```

---

## MODULE-LEVEL STATE (loaded once, reused)

```python
# Module-level singleton — model loaded once per process, never reloaded
_model: Optional[SentenceTransformer] = None
```

---

## FUNCTION 1: `load_model() -> SentenceTransformer`

**Signature is a CONTRACT. Never change it.**

```python
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
```

---

## FUNCTION 2: `compute_embedding(text: str, model: SentenceTransformer) -> np.ndarray`

**Signature is a CONTRACT. Never change it.**

```python
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
```

---

## FUNCTION 3: `embed_jd(jd: JobDescription, model: SentenceTransformer) -> JobDescription`

**Signature is a CONTRACT. Never change it.**

```python
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
```

---

## FUNCTION 4: `embed_candidates_batch(candidates: List[CandidateProfile], model: SentenceTransformer) -> List[CandidateProfile]`

**Signature is a CONTRACT. Never change it.**

```python
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
```

---

## FUNCTION 5: `cosine_similarity(a: np.ndarray, b: np.ndarray) -> float`

**Signature is a CONTRACT. Never change it.**

```python
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
```

---

## MODULE 2 TESTS — `tests/test_module2.py`

ALL 9 tests must pass before proceeding to Module 3.

```python
import numpy as np
import pytest
from src.nlp_engine import (
    load_model, compute_embedding, embed_jd,
    embed_candidates_batch, cosine_similarity,
)
from src.data_loader import load_candidates, get_jd


def test_model_loads_successfully():
    """Model loads without error."""
    model = load_model()
    assert model is not None


def test_model_singleton():
    """Calling load_model() twice returns the same object."""
    m1 = load_model()
    m2 = load_model()
    assert m1 is m2


def test_embedding_shape():
    """Single embedding returns shape (384,)."""
    model = load_model()
    emb = compute_embedding("Machine learning engineer with Python experience", model)
    assert emb.shape == (384,), f"Expected (384,) got {emb.shape}"


def test_empty_text_returns_zeros():
    """Empty string returns zero vector of shape (384,)."""
    model = load_model()
    emb = compute_embedding("", model)
    assert emb.shape == (384,)
    assert np.all(emb == 0.0)


def test_whitespace_text_returns_zeros():
    """Whitespace-only string returns zero vector."""
    model = load_model()
    emb = compute_embedding("   \n  \t  ", model)
    assert np.all(emb == 0.0)


def test_jd_embedding_set():
    """embed_jd sets jd.embedding to a numpy array."""
    model = load_model()
    jd = get_jd()
    assert jd.embedding is None
    jd = embed_jd(jd, model)
    assert jd.embedding is not None
    assert jd.embedding.shape == (384,)


def test_semantic_similarity_identical_text():
    """Identical texts produce similarity close to 1.0."""
    model = load_model()
    text = "Senior machine learning engineer with FAISS and Pinecone experience"
    e1 = compute_embedding(text, model)
    e2 = compute_embedding(text, model)
    sim = cosine_similarity(e1, e2)
    assert sim > 0.99, f"Identical text similarity should be >0.99, got {sim}"


def test_semantic_similarity_related_concepts():
    """Related AI concepts score higher than unrelated ones."""
    model = load_model()
    jd_emb = compute_embedding("vector database retrieval FAISS embeddings ranking", model)
    related_emb = compute_embedding("semantic search Pinecone dense retrieval", model)
    unrelated_emb = compute_embedding("graphic design photoshop branding typography", model)
    sim_related = cosine_similarity(jd_emb, related_emb)
    sim_unrelated = cosine_similarity(jd_emb, unrelated_emb)
    assert sim_related > sim_unrelated, \
        f"Related ({sim_related:.3f}) should beat unrelated ({sim_unrelated:.3f})"


def test_batch_embed_sets_embeddings():
    """embed_candidates_batch sets embedding on all candidates."""
    model = load_model()
    candidates = load_candidates("data/raw/candidates.jsonl")
    sample = candidates[:10]
    assert all(c.embedding is None for c in sample)
    result = embed_candidates_batch(sample, model)
    assert all(c.embedding is not None for c in result)
    assert all(c.embedding.shape == (384,) for c in result)


def test_cosine_similarity_zero_vector_safe():
    """cosine_similarity with zero vector returns 0.0 without error."""
    a = np.zeros(384, dtype=np.float32)
    b = np.ones(384, dtype=np.float32)
    sim = cosine_similarity(a, b)
    assert sim == 0.0
```

---

## COMPLETION CRITERIA

Module 2 is complete when:
1. `pytest tests/test_module2.py -v` shows ALL 9 tests passing
2. `embed_candidates_batch` on 5,000 candidates completes in under 100 seconds
3. Embedding model is confirmed loading from LOCAL cache (no network after pre-download)

Only then: proceed to MODULE_3_SCORING.md.
