"""
precompute.py — Run ONCE before rank.py.
Pre-computation may exceed the 5-minute ranking budget.
Only rank.py is time-constrained.

Usage:
    python precompute.py
"""
import subprocess
import sys
import time


def main():
    print("=" * 50)
    print("  Pre-computation — India Runs Track 1")
    print("  Run this once. Then run rank.py.")
    print("=" * 50)

    # Step 1: spaCy model
    print("\n[1/3] Downloading spaCy model...")
    subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        check=True,
    )
    print("      spaCy en_core_web_sm ready.")

    # Step 2: BGE embedding model (downloads to HuggingFace cache)
    print("\n[2/3] Downloading BAAI/bge-small-en-v1.5 (133 MB)...")
    print("      This requires network access — only needed once.")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    print(f"      Model cached in {time.time() - t0:.1f}s")

    # Step 3: Smoke test with sample candidates
    print("\n[3/3] Smoke test: loading sample candidates...")
    from src.data_loader import load_candidates, get_jd
    from src.nlp_engine import embed_jd, embed_candidates_batch

    sample = load_candidates("data/raw/sample_candidates.json")
    jd = get_jd()
    jd = embed_jd(jd, model)
    sample_embedded = embed_candidates_batch(sample[:5], model)
    print(f"      Loaded {len(sample)} sample candidates.")
    print(f"      Embedding shape: {sample_embedded[0].embedding.shape}")

    print("\n" + "=" * 50)
    print("  Pre-computation complete.")
    print("  You can now run (no network required):")
    print("  python rank.py --candidates ./candidates.jsonl --out ./PARTICIPANT_ID.csv")
    print("=" * 50)


if __name__ == "__main__":
    main()
