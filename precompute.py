"""
precompute.py — Pre-computation script (EXEMPT from 5-minute ranking limit)
Run ONCE before rank.py to download and cache:
  1. spaCy en_core_web_sm model
  2. BAAI/bge-small-en-v1.5 embedding model

These downloads can take several minutes on first run.
After this script succeeds, rank.py will complete in under 5 minutes.

Usage:
    python precompute.py
"""

import subprocess
import sys
from pathlib import Path


def download_spacy_model():
    print("=" * 60)
    print("Step 1/2: Downloading spaCy en_core_web_sm...")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        capture_output=False,   # show output live
    )
    if result.returncode != 0:
        raise RuntimeError("spaCy model download failed. Check your internet connection.")
    print("[SUCCESS] spaCy en_core_web_sm downloaded.\n")


def download_bge_model():
    print("=" * 60)
    print("Step 2/2: Downloading BAAI/bge-small-en-v1.5...")
    print("(~133MB — this may take 2-5 minutes on first run)")
    print("=" * 60)
    from sentence_transformers import SentenceTransformer
    from config import EMBEDDING_MODEL, DEVICE
    model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    model.eval()
    # Run a quick test encode to confirm the model works
    test_emb = model.encode("test sentence", convert_to_numpy=True, normalize_embeddings=True)
    assert test_emb.shape == (384,), f"Unexpected embedding shape: {test_emb.shape}"
    print(f"[SUCCESS] BGE model cached. Embedding shape: {test_emb.shape}\n")


def verify_environment():
    print("=" * 60)
    print("Verifying environment...")
    print("=" * 60)
    # Check models.py loads
    from src.models import CandidateProfile, CandidateScore
    print("  [OK] src.models OK")

    # Check data file exists
    candidates_path = Path("data/raw/candidates.jsonl")
    sample_path = Path("data/raw/sample_candidates.json")
    if candidates_path.exists():
        size_mb = candidates_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] candidates.jsonl found ({size_mb:.0f} MB)")
    else:
        print("  [WARNING] candidates.jsonl NOT found at data/raw/candidates.jsonl")
        print("       Judges will provide this file. Pipeline will work when present.")

    if sample_path.exists():
        print("  [OK] sample_candidates.json found (for Streamlit demo)")
    else:
        print("  [WARNING] sample_candidates.json not found — Streamlit demo needs it")

    print()


if __name__ == "__main__":
    print("\n[START] India Runs Track 1 — Pre-computation Script")
    print("This script caches all models. Run ONCE before rank.py.\n")

    download_spacy_model()
    download_bge_model()
    verify_environment()

    print("=" * 60)
    print("[SUCCESS] Pre-computation complete!")
    print("You can now run the ranking pipeline:")
    print("  python rank.py --candidates data/raw/candidates.jsonl --out submission.csv")
    print("=" * 60)
