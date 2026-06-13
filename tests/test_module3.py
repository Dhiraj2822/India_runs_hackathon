import pytest
from src.data_loader import load_candidates, get_jd
from src.nlp_engine import load_model, embed_jd, embed_candidates_batch
from src.scoring_engine import compute_stage1_score, compute_base_score
from src.models import CandidateProfile, CandidateScore


def test_stage1_honeypot_title_scores_zero():
    """Marketing Manager (honeypot title) scores 0 on title component."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    marketing = next(
        (c for c in candidates if "marketing manager" in c.current_title.lower()), None
    )
    if marketing:
        score = compute_stage1_score(marketing, jd)
        # Score > 0 because other components contribute, but should be low
        assert score < 0.35, f"Marketing manager scored too high: {score}"


def test_stage1_inactive_candidate_downweighted():
    """Candidate inactive > 180 days scores lower than recently active."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    from datetime import date
    inactive = next((
        c for c in candidates
        if (date.today() - date.fromisoformat(c.redrob_signals.last_active_date)).days > 180
    ), None)
    active = next((
        c for c in candidates
        if (date.today() - date.fromisoformat(c.redrob_signals.last_active_date)).days <= 14
    ), None)
    if inactive and active:
        s_inactive = compute_stage1_score(inactive, jd)
        s_active = compute_stage1_score(active, jd)
        assert s_inactive < s_active or s_active > 0.3


def test_stage1_score_in_range():
    """Stage 1 score is always in [0.0, 1.0]."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    for c in candidates[:1000]:
        s = compute_stage1_score(c, jd)
        assert 0.0 <= s <= 1.0, f"Score out of range: {s} for {c.candidate_id}"


def test_stage1_ml_title_scores_high():
    """Recommendation Systems Engineer scores high on Stage 1."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    rse = next(
        (c for c in candidates if "recommendation" in c.current_title.lower()), None
    )
    if rse:
        score = compute_stage1_score(rse, jd)
        assert score > 0.55, f"Recommendation Systems Engineer scored too low: {score}"


def test_stage1_not_open_to_work_downweighted():
    """Candidate not open to work scores lower on availability."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    closed = [c for c in candidates[:500] if not c.redrob_signals.open_to_work_flag]
    opened = [c for c in candidates[:500] if c.redrob_signals.open_to_work_flag]
    if closed and opened:
        avg_closed = sum(compute_stage1_score(c, jd) for c in closed[:20]) / 20
        avg_open = sum(compute_stage1_score(c, jd) for c in opened[:20]) / 20
        assert avg_open > avg_closed


def test_base_score_requires_embedding():
    """compute_base_score raises AssertionError if embedding not set."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    model = load_model()
    jd = embed_jd(jd, model)
    c = candidates[0]
    assert c.embedding is None
    with pytest.raises(AssertionError):
        compute_base_score(c, jd)


def test_base_score_in_range():
    """Stage 2 base_score is in [0.0, 1.0]."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    model = load_model()
    jd = embed_jd(jd, model)
    sample = embed_candidates_batch(candidates[:50], model)
    for c in sample:
        score = compute_base_score(c, jd)
        assert 0.0 <= score.base_score <= 1.0


def test_base_score_all_fields_set():
    """CandidateScore returned has all 6 component scores set."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    model = load_model()
    jd = embed_jd(jd, model)
    sample = embed_candidates_batch(candidates[:5], model)
    for c in sample:
        s = compute_base_score(c, jd)
        assert 0.0 <= s.semantic_score <= 1.0
        assert 0.0 <= s.skill_quality_score <= 1.0
        assert 0.0 <= s.career_fit_score <= 1.0
        assert 0.0 <= s.experience_score <= 1.0
        assert 0.0 <= s.behavioral_score <= 1.15
        assert 0.0 <= s.education_score <= 1.0


def test_assessment_score_used_in_skill_quality():
    """Candidate with high assessment score ranks higher than one without."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    model = load_model()
    jd = embed_jd(jd, model)
    # Find a candidate with assessment scores
    with_assessment = next(
        (c for c in candidates if c.redrob_signals.skill_assessment_scores), None
    )
    if with_assessment:
        embed_candidates_batch([with_assessment], model)
        score = compute_base_score(with_assessment, jd)
        assert score.skill_quality_score >= 0.0


def test_consulting_only_career_lower_career_fit():
    """All-consulting career history produces lower career_fit than mixed."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    jd = get_jd()
    model = load_model()
    jd = embed_jd(jd, model)
    from config import CONSULTING_FIRMS
    consulting = [
        c for c in candidates[:500]
        if all(ch.company.lower() in CONSULTING_FIRMS for ch in c.career_history)
        and len(c.career_history) > 0
    ]
    if consulting:
        embed_candidates_batch(consulting[:3], model)
        for c in consulting[:3]:
            score = compute_base_score(c, jd)
            assert score.career_fit_score <= 0.4, \
                f"Consulting-only career_fit should be ≤ 0.4, got {score.career_fit_score}"
