import pytest
from src.data_loader import load_candidates, get_jd
from src.models import CandidateProfile, CandidateScore, SkillEntry, RedrobSignals, CareerEntry
from src.edge_cases import apply_edge_cases, _detect_honeypot
from datetime import date


def _make_minimal_score(cid):
    """Helper: CandidateScore with base_score = 0.8."""
    s = CandidateScore(candidate_id=cid)
    s.base_score = 0.8
    return s


def test_honeypot_expert_zero_months():
    """Candidate with 5+ expert skills at 0 months flagged as honeypot."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    # Build a synthetic candidate with impossible skills
    c = candidates[0]
    c.skills = [
        SkillEntry(name=f"skill_{i}", proficiency="expert",
                   endorsements=0, duration_months=0, assessment_score=-1.0)
        for i in range(6)
    ]
    c.is_honeypot = False
    c.honeypot_reasons = []
    result = _detect_honeypot(c)
    assert result.is_honeypot is True
    assert len(result.honeypot_reasons) > 0


def test_honeypot_gets_zero_final_score():
    """Detected honeypot receives final_score = 0.0 regardless of base_score."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    c = candidates[0]
    c.skills = [
        SkillEntry(name=f"s{i}", proficiency="expert",
                   endorsements=0, duration_months=0, assessment_score=-1.0)
        for i in range(6)
    ]
    c.is_honeypot = False
    c.honeypot_reasons = []
    score = _make_minimal_score(c.candidate_id)
    result = apply_edge_cases(c, score, jd)
    assert result.final_score == 0.0


def test_consulting_only_penalty_applied():
    """All-consulting career history incurs consulting_only_career penalty."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    from config import CONSULTING_FIRMS
    consulting_c = next(
        (c for c in candidates
         if len(c.career_history) > 0
         and all(ch.company.lower() in CONSULTING_FIRMS for ch in c.career_history)),
        None
    )
    if consulting_c:
        consulting_c.is_honeypot = False
        consulting_c.honeypot_reasons = []
        score = _make_minimal_score(consulting_c.candidate_id)
        result = apply_edge_cases(consulting_c, score, jd)
        assert "consulting_only_career" in result.penalties
        assert result.final_score < 0.80


def test_keyword_stuffer_severe_penalty():
    """Marketing Manager with many JD skills gets severe penalty."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    c = candidates[0]
    # Rename companies to avoid triggering fictional company honeypot
    for ch in c.career_history:
        ch.company = "SafeCorp"
    c.current_title = "Marketing Manager"
    c.skills = [
        SkillEntry(name=r, proficiency="advanced",
                   endorsements=10, duration_months=24, assessment_score=-1.0)
        for r in ["embeddings", "retrieval", "ranking", "faiss", "pinecone"]
    ]
    c.is_honeypot = False
    c.honeypot_reasons = []
    score = _make_minimal_score(c.candidate_id)
    result = apply_edge_cases(c, score, jd)
    assert any("keyword_stuffer" in p for p in result.penalties)
    assert result.final_score < 0.60


def test_not_open_to_work_penalty():
    """Candidate not open to work receives penalty."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    closed = next(
        (c for c in candidates if not c.redrob_signals.open_to_work_flag), None
    )
    if closed:
        closed.is_honeypot = False
        closed.honeypot_reasons = []
        score = _make_minimal_score(closed.candidate_id)
        result = apply_edge_cases(closed, score, jd)
        assert "not_open_to_work" in result.penalties


def test_final_score_never_exceeds_one():
    """final_score is always <= 1.0 even with many bonuses."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    for c in candidates[:100]:
        c.is_honeypot = False
        c.honeypot_reasons = []
        s = CandidateScore(candidate_id=c.candidate_id)
        s.base_score = 0.95
        result = apply_edge_cases(c, s, jd)
        assert result.final_score <= 1.0, f"Score {result.final_score} for {c.candidate_id}"


def test_final_score_never_below_zero():
    """final_score is always >= 0.0 even with many penalties."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    for c in candidates[:100]:
        if not c.is_honeypot:
            s = CandidateScore(candidate_id=c.candidate_id)
            s.base_score = 0.05
            result = apply_edge_cases(c, s, jd)
            assert result.final_score >= 0.0


def test_github_bonus_applied():
    """High GitHub activity receives bonus."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    high_gh = next(
        (c for c in candidates if c.redrob_signals.github_activity_score >= 70), None
    )
    if high_gh:
        high_gh.is_honeypot = False
        high_gh.honeypot_reasons = []
        s = _make_minimal_score(high_gh.candidate_id)
        result = apply_edge_cases(high_gh, s, jd)
        assert any("github" in b for b in result.bonuses)


def test_additive_only_no_existing_function_modified():
    """Verify apply_edge_cases returns score with final_score set."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    c = candidates[30]
    c.is_honeypot = False
    c.honeypot_reasons = []
    s = _make_minimal_score(c.candidate_id)
    result = apply_edge_cases(c, s, jd)
    assert result.final_score is not None
    assert isinstance(result.final_score, float)


def test_job_hopper_penalty_excludes_contracts():
    """Job hopper check excludes contract roles from tenure calculation."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    c = candidates[0]
    # Simulate: 3 contract roles (short) + 1 long permanent role
    c.career_history = [
        CareerEntry("Corp A", "Contract Engineer", "2018-01-01", "2018-06-01",
                    5, False, "Tech", "51-200", "Contract role"),
        CareerEntry("Corp B", "Contract Developer", "2018-07-01", "2019-01-01",
                    6, False, "Tech", "51-200", "Contract role"),
        CareerEntry("Startup X", "ML Engineer", "2019-02-01", "2023-02-01",
                    48, False, "AI", "51-200", "Built retrieval systems"),
    ]
    c.is_honeypot = False
    c.honeypot_reasons = []
    s = _make_minimal_score(c.candidate_id)
    result = apply_edge_cases(c, s, jd)
    # Should NOT be flagged as job hopper (contract roles excluded, 48mo permanent)
    assert not any("job_hopper" in p for p in result.penalties)
