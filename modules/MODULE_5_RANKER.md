# MODULE 5 — RANKER & PIPELINE ORCHESTRATOR
## File to build: `src/ranker.py`
## Depends on: ALL previous modules (1-4) + `src/reasoning.py` (Module 6)
## Do NOT start until ALL 10 Module 4 tests pass.
## NOTE: Module 6 (reasoning.py) must be built before rank.py can run end-to-end.
##       Build Module 5 first, then Module 6, then wire rank.py.

---

## WHAT THIS MODULE DOES

This is the pipeline orchestrator. It calls every other module in the correct order
and produces the final `submission.csv`. It contains no scoring logic of its own —
only sequencing, sorting, tie-breaking, and output writing.

Two public functions:
- `rank_all()` — runs the full 2-stage pipeline, returns top 100 `CandidateScore` objects
- `write_submission()` — writes CSV and runs the hackathon validator

---

## IMPORTS

```python
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
```

---

## FUNCTION 1: `rank_all(candidates, jd, model) -> List[CandidateScore]`

**Signature is a CONTRACT. Never change it.**

```python
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

    # ── Sort by final_score desc; tie-break: candidate_id ascending ───────────
    # Tie-break rule from submission_spec.docx:
    # When final_score is equal, lower candidate_id (alphabetically) gets lower rank number
    scored_pairs.sort(key=lambda x: (-x[1].final_score, x[0].candidate_id))

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
```

---

## FUNCTION 2: `write_submission(scores, candidates_map, output_path) -> None`

**Signature is a CONTRACT. Never change it.**

```python
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

    # ── Write CSV ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
    df = df.sort_values("rank").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"  Written: {output_path}")

    # ── Run the hackathon validator ───────────────────────────────────────────
    validator_path = Path("validate_submission.py")
    if not validator_path.exists():
        print("  ⚠️  validate_submission.py not found — skipping validation")
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
    print(f"  ✅ Submission valid: {result.stdout.strip()}")
```

---

## MODULE 5 TESTS — `tests/test_module5.py`

ALL 9 tests must pass before proceeding to Module 6.

```python
import os
import tempfile
import pytest
import pandas as pd
from src.data_loader import load_candidates, get_jd
from src.nlp_engine import load_model, embed_jd
from src.ranker import rank_all, write_submission
from config import TOP_K_FINAL


@pytest.fixture(scope="module")
def pipeline_output():
    """Run the full pipeline once and cache results for all tests in this module."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    model = load_model()
    jd = get_jd()
    jd = embed_jd(jd, model)
    scores = rank_all(candidates, jd, model)
    candidates_map = {c.candidate_id: c for c in candidates}
    return scores, candidates_map


def test_output_count(pipeline_output):
    """Pipeline returns exactly 100 scored candidates."""
    scores, _ = pipeline_output
    assert len(scores) == TOP_K_FINAL, f"Expected 100, got {len(scores)}"


def test_ranks_are_unique(pipeline_output):
    """All ranks from 1 to 100 are present exactly once."""
    scores, _ = pipeline_output
    ranks = [s.rank for s in scores]
    assert sorted(ranks) == list(range(1, TOP_K_FINAL + 1))


def test_scores_non_increasing(pipeline_output):
    """Scores are non-increasing from rank 1 to rank 100."""
    scores, _ = pipeline_output
    sorted_scores = sorted(scores, key=lambda s: s.rank)
    for i in range(len(sorted_scores) - 1):
        assert sorted_scores[i].final_score >= sorted_scores[i + 1].final_score - 1e-9, \
            f"Score at rank {i+1} ({sorted_scores[i].final_score}) < rank {i+2} ({sorted_scores[i+1].final_score})"


def test_scores_in_valid_range(pipeline_output):
    """All final_scores are in [0.0, 1.0]."""
    scores, _ = pipeline_output
    for s in scores:
        assert 0.0 <= s.final_score <= 1.0, \
            f"Score {s.final_score} out of range for {s.candidate_id}"


def test_no_honeypot_in_top_100(pipeline_output):
    """No honeypot candidate appears in top 100."""
    scores, candidates_map = pipeline_output
    for s in scores:
        c = candidates_map[s.candidate_id]
        assert not c.is_honeypot, \
            f"Honeypot {c.candidate_id} in top 100 at rank {s.rank}"


def test_reasoning_not_empty(pipeline_output):
    """Every candidate has a non-empty reasoning string."""
    scores, _ = pipeline_output
    for s in scores:
        assert s.reasoning and len(s.reasoning) > 10, \
            f"Empty reasoning for rank {s.rank} ({s.candidate_id})"


def test_reasoning_all_different(pipeline_output):
    """At least 90 of 100 reasoning strings are unique (prevents pure templating)."""
    scores, _ = pipeline_output
    unique_reasonings = len(set(s.reasoning for s in scores))
    assert unique_reasonings >= 90, \
        f"Only {unique_reasonings}/100 unique reasoning strings — too templated"


def test_write_submission_creates_valid_csv(pipeline_output, tmp_path):
    """write_submission creates a CSV that passes the hackathon validator."""
    scores, candidates_map = pipeline_output
    out = str(tmp_path / "test_submission.csv")
    write_submission(scores, candidates_map, out)
    assert os.path.exists(out)
    df = pd.read_csv(out)
    assert list(df.columns) == ["candidate_id", "rank", "score", "reasoning"]
    assert len(df) == TOP_K_FINAL


def test_tie_break_by_candidate_id(pipeline_output):
    """Among tied final_scores, lower candidate_id has lower rank number."""
    scores, _ = pipeline_output
    sorted_scores = sorted(scores, key=lambda s: s.rank)
    for i in range(len(sorted_scores) - 1):
        s1, s2 = sorted_scores[i], sorted_scores[i + 1]
        if abs(s1.final_score - s2.final_score) < 1e-9:
            # Tied — lower rank number should have alphabetically lower candidate_id
            assert s1.candidate_id <= s2.candidate_id, \
                f"Tie-break violated: rank {s1.rank} ({s1.candidate_id}) vs rank {s2.rank} ({s2.candidate_id})"
```

---

## COMPLETION CRITERIA

Module 5 is complete when:
1. `pytest tests/test_module5.py -v` shows ALL 9 tests passing
2. No honeypot appears in the top 100
3. All reasoning strings are unique (≥ 90 of 100)
4. The validator prints `"Submission is valid."`

Only then: proceed to MODULE_6_REASONING.md.
