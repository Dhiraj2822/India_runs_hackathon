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
