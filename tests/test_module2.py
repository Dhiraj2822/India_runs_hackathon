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
