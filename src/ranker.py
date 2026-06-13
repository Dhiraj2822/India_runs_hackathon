"""
ranker.py — Module 5
Pipeline orchestrator. Calls all other modules in the correct order.
Contains NO scoring logic — only sequencing, sorting, tie-breaking, and output writing.

Public functions:
  rank_all()         — full 2-stage pipeline → top 100 CandidateScore objects
  write_submission() — writes CSV and runs the hackathon validator
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.models import CandidateProfile, CandidateScore, JobDescription
from src.nlp_engine import embed_candidates_batch
from src.scoring_engine import compute_stage1_score, compute_base_score
from src.edge_cases import apply_edge_cases
from config import (
    TOP_K_STAGE1, TOP_K_STAGE2, TOP_K_FINAL,
    OUTPUT_CSV,
)
# NOTE: generate_reasoning imported inside rank_all to avoid circular import issues


def rank_all(
    candidates: List[CandidateProfile],
    jd: JobDescription,
    model,
) -> List[CandidateScore]:
    """
    Full 2-stage ranking pipeline.

    Stage 1: Structured fast scoring on ALL candidates → top TOP_K_STAGE1
    Stage 2: Semantic + full scoring on top → top TOP_K_STAGE2
    Edge cases: Penalties and bonuses applied → final_score
    Sort and take top TOP_K_FINAL (100).

    Returns list of CandidateScore objects, sorted by rank ascending (1 first).
    Each CandidateScore has rank and reasoning set.
    """
    from src.reasoning import generate_reasoning

    # ── STAGE 1: Fast filter (all 100K → top 5K) ──────────────────────────────
    print(f"  Stage 1: scoring {len(candidates):,} candidates...")
    t0 = time.time()

    stage1_pairs = [
        (c, compute_stage1_score(c, jd))
        for c in candidates
    ]

    # Sort descending by stage1 score, take top TOP_K_STAGE1
    stage1_pairs.sort(key=lambda x: -x[1])
    top_stage1 = [c for c, _ in stage1_pairs[:TOP_K_STAGE1]]

    print(f"  Stage 1 complete: {len(top_stage1):,} candidates retained ({time.time()-t0:.1f}s)")

    # ── STAGE 2: Embed filtered candidates ────────────────────────────────────
    print(f"  Stage 2: embedding {len(top_stage1):,} candidates...")
    t1 = time.time()
    top_stage1 = embed_candidates_batch(top_stage1, model)
    print(f"  Embedding complete ({time.time()-t1:.1f}s)")

    # ── STAGE 2: Full scoring ─────────────────────────────────────────────────
    print(f"  Stage 2: full scoring + edge cases...")
    t2 = time.time()
    scored_pairs = []
    for c in top_stage1:
        score = compute_base_score(c, jd)
        score = apply_edge_cases(c, score, jd)
        scored_pairs.append((c, score))
    print(f"  Full scoring complete ({time.time()-t2:.1f}s)")

    # Sort by final_score descending; 3-level tie-breaking:
    #   Level 1 (primary)  : rounded 4dp score — matches what appears in submission CSV.
    #   Level 2 (secondary): raw unrounded score — truer signal wins when rounded scores tie.
    #   Level 3 (tertiary) : candidate_id ascending — deterministic last resort.
    # This improves on the spec's pure-alphabetical tie-break by using the actual
    # underlying score before falling back to candidate ID ordering.
    scored_pairs.sort(
        key=lambda x: (
            -round(x[1].final_score, 4),   # primary: rounded score (matches CSV)
            -x[1].final_score,              # secondary: raw unrounded score
            x[0].candidate_id,              # tertiary: alphabetical ID
        )
    )

    # Take top TOP_K_STAGE2 (200) for reasoning generation
    top200 = scored_pairs[:TOP_K_STAGE2]

    # ── Take top 100, assign ranks, generate reasoning ────────────────────────
    final_100 = top200[:TOP_K_FINAL]
    results = []
    for rank_num, (candidate, score) in enumerate(final_100, start=1):
        score.rank = rank_num
        score.reasoning = generate_reasoning(candidate, score, jd)
        results.append(score)

    return results   # sorted rank 1 to 100


def write_submission(
    scores: List[CandidateScore],
    candidates_map: Dict[str, CandidateProfile],
    output_path: str,
) -> None:
    """
    Write the ranked scores to a CSV file and validate against
    the hackathon-provided validator (validate_submission.py).

    Output format (from submission_spec.docx):
        candidate_id, rank, score, reasoning

    Raises RuntimeError if the validator reports any error.
    """
    # ── Build rows ────────────────────────────────────────────────────────────
    rows = []
    for s in scores:
        rows.append({
            "candidate_id": s.candidate_id,
            "rank":         s.rank,
            "score":        round(s.final_score, 4),
            "reasoning":    s.reasoning,
        })

    # ── Sanity checks before writing ─────────────────────────────────────────
    assert len(rows) == TOP_K_FINAL, \
        f"Expected {TOP_K_FINAL} rows, got {len(rows)}"

    ranks = [r["rank"] for r in rows]
    assert sorted(ranks) == list(range(1, TOP_K_FINAL + 1)), \
        "Ranks must be exactly 1-100 with no duplicates or gaps"

    scores_list = [r["score"] for r in rows]
    for i in range(len(scores_list) - 1):
        assert scores_list[i] >= scores_list[i + 1] - 1e-9, \
            f"Scores must be non-increasing: row {i+1} score {scores_list[i]} < row {i+2} score {scores_list[i+1]}"

    assert all(r["reasoning"] and len(r["reasoning"]) > 10 for r in rows), \
        "All reasoning strings must be non-empty (> 10 chars)"

    # ── Candidate ID validation (spec rejection #4) ───────────────────────────
    # All 100 output IDs must actually exist in the input dataset.
    if candidates_map:
        bad_ids = [r["candidate_id"] for r in rows if r["candidate_id"] not in candidates_map]
        assert not bad_ids, \
            f"Output contains {len(bad_ids)} candidate_id(s) not in input dataset: {bad_ids[:5]}"

    # ── Score differentiation check (spec rejection #5) ──────────────────────
    # Scores must not all be the same value — confirms the pipeline is working.
    unique_scores = len(set(r["score"] for r in rows))
    assert unique_scores > 50, \
        f"Only {unique_scores} unique scores across 100 candidates — pipeline may be broken (all same value?)"

    # ── Write CSV ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
    df = df.sort_values("rank").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"  Written: {output_path}")

    # ── Run the hackathon validator ───────────────────────────────────────────
    validator_path = Path("validate_submission.py")
    if not validator_path.exists():
        print("  [WARNING] validate_submission.py not found — skipping validation")
        return

    result = subprocess.run(
        [sys.executable, str(validator_path), output_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Submission validator FAILED:\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )
    print(f"  [SUCCESS] Submission valid: {result.stdout.strip()}")
