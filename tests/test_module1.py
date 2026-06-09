import pytest
import json
from src.data_loader import load_candidates, get_jd
from src.models import CandidateProfile, JobDescription, RedrobSignals


def test_first_candidate_id():
    """First candidate in file is CAND_0000001."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    assert candidates[0].candidate_id == "CAND_0000001"


def test_total_candidate_count():
    """All 100,000 candidates load successfully."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    assert len(candidates) == 100_000


def test_skills_are_lowercase():
    """All skill names are lowercased."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    for c in candidates[:200]:
        for s in c.skills:
            assert s.name == s.name.lower(), f"Skill not lowercase: {s.name}"


def test_proficiency_valid_values():
    """All proficiency values are in the allowed enum."""
    valid = {"beginner", "intermediate", "advanced", "expert"}
    candidates = load_candidates("data/raw/candidates.jsonl")
    for c in candidates[:500]:
        for s in c.skills:
            assert s.proficiency in valid, f"Invalid proficiency: {s.proficiency}"


def test_assessment_scores_injected():
    """Assessment scores are injected into matching SkillEntry objects."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    # CAND_0000001 has NLP assessment 38.8 in redrob_signals
    c = next(x for x in candidates if x.candidate_id == "CAND_0000001")
    nlp_skill = next((s for s in c.skills if s.name == "nlp"), None)
    assert nlp_skill is not None, "NLP skill not found in CAND_0000001"
    assert abs(nlp_skill.assessment_score - 38.8) < 0.01, \
        f"Expected 38.8 got {nlp_skill.assessment_score}"


def test_assessment_scores_lowercase_match():
    """Assessment score dict keys are lowercased to match skill names."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    c = candidates[0]
    for skill_name in c.redrob_signals.skill_assessment_scores.keys():
        assert skill_name == skill_name.lower(), f"Key not lowercased: {skill_name}"


def test_full_text_not_empty():
    """full_text is populated for every candidate."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    assert all(len(c.full_text) > 20 for c in candidates[:500])


def test_redrob_signals_response_rate_range():
    """recruiter_response_rate is between 0.0 and 1.0."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    for c in candidates[:500]:
        assert 0.0 <= c.redrob_signals.recruiter_response_rate <= 1.0


def test_education_tier_valid():
    """Education tier values are in the allowed enum."""
    valid_tiers = {"tier_1", "tier_2", "tier_3", "tier_4", "unknown"}
    candidates = load_candidates("data/raw/candidates.jsonl")
    for c in candidates[:500]:
        for e in c.education:
            assert e.tier in valid_tiers, f"Invalid tier: {e.tier}"


def test_malformed_line_skipped_no_crash():
    """A malformed JSON line is skipped without crashing."""
    import tempfile, os
    good_line = json.dumps({
        "candidate_id": "CAND_9999999",
        "profile": {"anonymized_name": "Test", "headline": "", "summary": "",
                    "location": "", "country": "", "years_of_experience": 3.0,
                    "current_title": "Engineer", "current_company": "Test Co",
                    "current_company_size": "11-50", "current_industry": "Tech"},
        "career_history": [], "education": [], "skills": [],
        "certifications": [], "languages": [],
        "redrob_signals": {
            "profile_completeness_score": 50.0, "signup_date": "2024-01-01",
            "last_active_date": "2026-01-01", "open_to_work_flag": True,
            "profile_views_received_30d": 0, "applications_submitted_30d": 0,
            "recruiter_response_rate": 0.5, "avg_response_time_hours": 24.0,
            "skill_assessment_scores": {}, "connection_count": 10,
            "endorsements_received": 0, "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 10.0, "max": 20.0},
            "preferred_work_mode": "hybrid", "willing_to_relocate": True,
            "github_activity_score": -1, "search_appearance_30d": 5,
            "saved_by_recruiters_30d": 0, "interview_completion_rate": 0.8,
            "offer_acceptance_rate": -1, "verified_email": True,
            "verified_phone": True, "linkedin_connected": False
        }
    })
    bad_line = "{ this is not valid json }"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(good_line + '\n')
        f.write(bad_line + '\n')
        tmp_path = f.name
    result = load_candidates(tmp_path)
    os.unlink(tmp_path)
    assert len(result) == 1
    assert result[0].candidate_id == "CAND_9999999"


def test_get_jd_returns_correct_type():
    """get_jd returns a JobDescription with populated fields."""
    jd = get_jd()
    assert isinstance(jd, JobDescription)
    assert len(jd.required_skills) > 0
    assert jd.experience_min == 5.0
    assert jd.experience_max == 9.0
    assert jd.embedding is None  # set later by nlp_engine


def test_current_title_preserved():
    """current_title is stored without modification (case preserved)."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    c = next(x for x in candidates if x.candidate_id == "CAND_0000001")
    assert c.current_title == "Backend Engineer"
