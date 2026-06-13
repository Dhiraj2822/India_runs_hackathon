# MODULE 1 — DATA LOADER
## File to build: `src/data_loader.py`
## Depends on: `src/models.py` (must exist first)
## ─────────────────────────────────────────────────────────────────────────────

## WHAT THIS MODULE DOES

Reads `candidates.jsonl` (100,000 lines) and converts every line into a
`CandidateProfile` object. Also returns the `JobDescription` object (no file
reading needed — the JD text is embedded in `config.py`).

This module has NO scoring logic. No embeddings. No edge case detection.
It only parses and validates the raw data.

---

## IMPORTS

```python
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.models import (
    CandidateProfile, SkillEntry, CareerEntry, EducationEntry,
    RedrobSignals, JobDescription, CandidateScore
)
from config import (
    JD_TEXT, JD_REQUIRED_SKILLS, JD_PREFERRED_SKILLS,
    JD_EXPERIENCE_MIN, JD_EXPERIENCE_MAX, JD_PREFERRED_LOCATIONS,
    CONSULTING_FIRMS, HONEYPOT_TITLES, LOGS_PATH
)
```

---

## FUNCTION 1: `load_candidates(path: str) -> List[CandidateProfile]`

**Signature is a CONTRACT. Never change it.**

```python
def load_candidates(path: str) -> List[CandidateProfile]:
    """
    Load candidates.jsonl file. One JSON object per line.
    Returns list of CandidateProfile objects.
    Skips and logs malformed lines — never raises on bad input.
    """
```

**Implementation steps:**

```
1. Open file with UTF-8 encoding
2. For each non-empty line:
   a. json.loads(line.strip())
   b. Call _parse_candidate(raw_dict)
   c. If result is not None → append to list
   d. If any exception → log warning with line number, continue
3. Log total loaded and total skipped
4. Return list
```

**Error handling:**
- `json.JSONDecodeError` on a line → log warning, skip, continue
- `KeyError` / `ValueError` in `_parse_candidate` → log warning with candidate_id if available, skip, continue
- Never raise any exception that stops the pipeline

---

## FUNCTION 2: `_parse_candidate(raw: dict) -> Optional[CandidateProfile]` *(private)*

**Returns None only if candidate_id is missing or malformed.**

```
Step 1 — candidate_id
  cid = raw.get('candidate_id', '').strip()
  Validate format: must match r'^CAND_[0-9]{7}$'
  If invalid → return None

Step 2 — profile block
  p = raw['profile']
  name = str(p.get('anonymized_name', '')).strip()
  headline = str(p.get('headline', '')).strip()
  summary = str(p.get('summary', '')).strip()
  location = str(p.get('location', '')).strip()
  country = str(p.get('country', '')).strip()
  years_of_experience = float(p.get('years_of_experience', 0.0))
  current_title = str(p.get('current_title', '')).strip()
  current_company = str(p.get('current_company', '')).strip()
  current_company_size = str(p.get('current_company_size', '')).strip()
  current_industry = str(p.get('current_industry', '')).strip()

Step 3 — skills
  For each item in raw.get('skills', []):
    name = str(item['name']).strip().lower()
    proficiency = str(item.get('proficiency', 'beginner'))
    If proficiency not in VALID_PROFICIENCIES → proficiency = 'beginner'
    endorsements = max(0, int(item.get('endorsements', 0)))
    duration_months = max(0, int(item.get('duration_months', 0)))
    assessment_score = -1.0   ← filled in Step 7
    Append SkillEntry(name, proficiency, endorsements, duration_months, assessment_score)

  VALID_PROFICIENCIES = {'beginner', 'intermediate', 'advanced', 'expert'}

Step 4 — career_history
  For each item in raw.get('career_history', []):
    company = str(item.get('company', '')).strip()
    title = str(item.get('title', '')).strip()
    start_date = str(item.get('start_date', '')).strip()
    end_date = item.get('end_date')  ← keep as None if null
    duration_months = max(0, int(item.get('duration_months', 0)))
    is_current = bool(item.get('is_current', False))
    industry = str(item.get('industry', '')).strip()
    company_size = str(item.get('company_size', '')).strip()
    description = str(item.get('description', '')).strip()
    Append CareerEntry(...)

Step 5 — education
  For each item in raw.get('education', []):
    institution = str(item.get('institution', '')).strip()
    degree = str(item.get('degree', '')).strip()
    field_of_study = str(item.get('field_of_study', '')).strip()
    start_year = int(item.get('start_year', 0))
    end_year = int(item.get('end_year', 0))
    grade = item.get('grade')  ← keep None if null/missing
    tier = str(item.get('tier', 'unknown'))
    If tier not in VALID_TIERS → tier = 'unknown'
    Append EducationEntry(...)

  VALID_TIERS = {'tier_1', 'tier_2', 'tier_3', 'tier_4', 'unknown'}

Step 6 — redrob_signals
  sig = raw.get('redrob_signals', {})
  Parse every field with safe defaults:

  profile_completeness_score = float(sig.get('profile_completeness_score', 0.0))
  signup_date = str(sig.get('signup_date', '2000-01-01'))
  last_active_date = str(sig.get('last_active_date', '2000-01-01'))
  open_to_work_flag = bool(sig.get('open_to_work_flag', False))
  profile_views_received_30d = max(0, int(sig.get('profile_views_received_30d', 0)))
  applications_submitted_30d = max(0, int(sig.get('applications_submitted_30d', 0)))
  recruiter_response_rate = float(sig.get('recruiter_response_rate', 0.0))
  avg_response_time_hours = float(sig.get('avg_response_time_hours', 999.0))
  skill_assessment_scores = {
      str(k).lower(): float(v)
      for k, v in sig.get('skill_assessment_scores', {}).items()
  }
  ← NOTE: keys lowercased to match SkillEntry.name

  connection_count = max(0, int(sig.get('connection_count', 0)))
  endorsements_received = max(0, int(sig.get('endorsements_received', 0)))
  notice_period_days = max(0, int(sig.get('notice_period_days', 90)))
  salary_range = sig.get('expected_salary_range_inr_lpa', {})
  expected_salary_min_lpa = float(salary_range.get('min', 0.0))
  expected_salary_max_lpa = float(salary_range.get('max', 0.0))
  preferred_work_mode = str(sig.get('preferred_work_mode', 'flexible'))
  willing_to_relocate = bool(sig.get('willing_to_relocate', False))
  github_activity_score = float(sig.get('github_activity_score', -1.0))
  search_appearance_30d = max(0, int(sig.get('search_appearance_30d', 0)))
  saved_by_recruiters_30d = max(0, int(sig.get('saved_by_recruiters_30d', 0)))
  interview_completion_rate = float(sig.get('interview_completion_rate', 0.5))
  offer_acceptance_rate = float(sig.get('offer_acceptance_rate', -1.0))
  verified_email = bool(sig.get('verified_email', False))
  verified_phone = bool(sig.get('verified_phone', False))
  linkedin_connected = bool(sig.get('linkedin_connected', False))

  Create RedrobSignals(all fields above)

Step 7 — inject assessment_scores into SkillEntry objects
  For each skill in skills:
    lowered_name = skill.name  ← already lowercased in Step 3
    If lowered_name in redrob_signals.skill_assessment_scores:
      skill.assessment_score = redrob_signals.skill_assessment_scores[lowered_name]

Step 8 — build full_text for embedding
  Concatenate with single spaces:
  parts = [
    current_title,
    headline,
    summary,
    ' '.join(s.name for s in skills),
    ' '.join(f"{c.title} {c.description}" for c in career_history),
    ' '.join(f"{e.degree} {e.field_of_study}" for e in education),
  ]
  full_text = ' '.join(p for p in parts if p.strip())

Step 9 — certifications and languages (keep as raw dicts)
  certifications = list(raw.get('certifications', []))
  languages = list(raw.get('languages', []))

Step 10 — Return CandidateProfile
  Return CandidateProfile(
    candidate_id=cid,
    name=name,
    headline=headline,
    summary=summary,
    location=location,
    country=country,
    years_of_experience=years_of_experience,
    current_title=current_title,
    current_company=current_company,
    current_company_size=current_company_size,
    current_industry=current_industry,
    skills=skills,
    career_history=career_history,
    education=education,
    certifications=certifications,
    languages=languages,
    redrob_signals=redrob_signals,
    full_text=full_text,
    embedding=None,
    is_honeypot=False,
    honeypot_reasons=[],
  )
```

---

## FUNCTION 3: `get_jd() -> JobDescription`

**Signature is a CONTRACT. Never change it.**

```python
def get_jd() -> JobDescription:
    """
    Return the JobDescription for this challenge.
    JD content is embedded in config.py — no file reading needed at ranking time.
    """
    return JobDescription(
        raw_text=JD_TEXT,
        required_skills=JD_REQUIRED_SKILLS,
        preferred_skills=JD_PREFERRED_SKILLS,
        experience_min=JD_EXPERIENCE_MIN,
        experience_max=JD_EXPERIENCE_MAX,
        seniority_level="senior",
        domain="ai_ml",
        preferred_locations=JD_PREFERRED_LOCATIONS,
        disqualified_titles=list(HONEYPOT_TITLES),
        disqualified_companies=list(CONSULTING_FIRMS),
        embedding=None,
    )
```

---

## HELPER: `_log_warning(message: str)` *(private)*

```python
import logging
from config import LOGS_PATH

logging.basicConfig(
    filename=LOGS_PATH,
    level=logging.WARNING,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

def _log_warning(message: str):
    logging.warning(message)
```

---

## MODULE 1 TESTS — `tests/test_module1.py`

ALL 12 tests must pass. Zero failures before proceeding to Module 2.

```python
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
```

---

## COMPLETION CRITERIA

Module 1 is complete when:
1. `pytest tests/test_module1.py -v` shows ALL 12 tests passing
2. No warnings in `logs/errors.log` for the first 1,000 candidates
3. Runtime for loading all 100K candidates is under 30 seconds

Only then: proceed to MODULE_2_NLP.md.
