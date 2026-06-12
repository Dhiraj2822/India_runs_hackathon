"""
rank.py — Main CLI entry point for India Runs Track 1.
Performs the ranking pipeline on an input candidates JSONL file.
Pre-cached models will be loaded automatically (run precompute.py first).

Usage:
    python rank.py --candidates <path_to_jsonl> --out <path_to_csv>
"""

import argparse
import sys
import time
from pathlib import Path

from config import OUTPUT_CSV, CANDIDATES_JSONL
from src.data_loader import load_candidates, get_jd
from src.nlp_engine import load_model, embed_jd
from src.ranker import rank_all, write_submission


def main():
    parser = argparse.ArgumentParser(description="India Runs Track 1 — Ranking CLI")
    parser.add_argument(
        "--candidates",
        type=str,
        default=CANDIDATES_JSONL,
        help=f"Path to candidates.jsonl input file (default: {CANDIDATES_JSONL})"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=OUTPUT_CSV,
        help=f"Path to output submission CSV file (default: {OUTPUT_CSV})"
    )
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"[ERROR] Candidates file not found at: {candidates_path}")
        sys.exit(1)

    print(f"Loading candidates from: {candidates_path}...")
    t0 = time.time()
    candidates = load_candidates(str(candidates_path))
    print(f"Loaded {len(candidates):,} candidates in {time.time() - t0:.2f}s")

    if not candidates:
        print("[ERROR] No candidates were successfully loaded.")
        sys.exit(1)

    print("Loading model and embedding Job Description...")
    t1 = time.time()
    model = load_model()
    jd = get_jd()
    jd = embed_jd(jd, model)
    print(f"Model and JD loaded in {time.time() - t1:.2f}s")

    print("Running ranking pipeline...")
    t2 = time.time()
    scores = rank_all(candidates, jd, model)
    print(f"Ranking pipeline finished in {time.time() - t2:.2f}s")

    # Map candidate IDs for validation
    candidates_map = {c.candidate_id: c for c in candidates}

    # Ensure output directory exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing submission to: {out_path}...")
    try:
        write_submission(scores, candidates_map, str(out_path))
        print("*** Pipeline completed successfully! ***")
    except Exception as e:
        print(f"[ERROR] Error writing or validating submission: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
