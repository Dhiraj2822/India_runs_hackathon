"""
data_loader.py — Module 1
Reads candidates.jsonl and returns List[CandidateProfile].
Also returns the JobDescription (embedded in config.py — no file read needed).
NO scoring logic. NO embeddings. NO edge case detection. Parse and validate only.
"""

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

# ─── Logging setup ────────────────────────────────────────────────────────────
Path(LOGS_PATH).parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOGS_PATH,
    level=logging.WARNING,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

# ─── Valid enum sets ───────────────────────────────────────────────────────────
VALID_PROFICIENCIES = {'beginner', 'intermediate', 'advanced', 'expert'}
VALID_TIERS = {'tier_1', 'tier_2', 'tier_3', 'tier_4', 'unknown'}
_CAND_ID_PATTERN = re.compile(r'^CAND_[0-9]{7}$')


def _log_warning(message: str) -> None:
    logging.warning(message)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(path: str) -> List[CandidateProfile]:
    """
    Load candidates.jsonl file. One JSON object per line.
    Returns list of CandidateProfile objects.
    Skips and logs malformed lines — never raises on bad input.
    """
    candidates: List[CandidateProfile] = []
    skipped = 0

    with open(path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                result = _parse_candidate(raw)
                if result is not None:
                    candidates.append(result)
                else:
                    skipped += 1
                    _log_warning(f"line={line_num} | skipped: invalid or missing candidate_id")
            except json.JSONDecodeError as e:
                skipped += 1
                _log_warning(f"line={line_num} | JSONDecodeError: {e}")
            except Exception as e:
                skipped += 1
                cid = None
                try:
                    cid = raw.get('candidate_id', 'unknown') if isinstance(raw, dict) else 'unknown'
                except Exception:
                    cid = 'unknown'
                _log_warning(f"line={line_num} | candidate_id={cid} | Error: {e}")

    _log_warning(f"load_candidates complete: loaded={len(candidates)}, skipped={skipped}")
    return candidates


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
        preferred_locations=list(JD_PREFERRED_LOCATIONS),
        disqualified_titles=list(HONEYPOT_TITLES),
        disqualified_companies=list(CONSULTING_FIRMS),
        embedding=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE PARSING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_candidate(raw: dict) -> Optional[CandidateProfile]:
    """
    Parse one raw JSON dict into a CandidateProfile.
    Returns None only if candidate_id is missing or malformed.
    """
    # Step 1 — candidate_id
    cid = str(raw.get('candidate_id', '')).strip()
    if not _CAND_ID_PATTERN.match(cid):
        return None

    # Step 2 — profile block
    p = raw['profile']
    name                 = str(p.get('anonymized_name', '')).strip()
    headline             = str(p.get('headline', '')).strip()
    summary              = str(p.get('summary', '')).strip()
    location             = str(p.get('location', '')).strip()
    country              = str(p.get('country', '')).strip()
    years_of_experience  = float(p.get('years_of_experience', 0.0))
    current_title        = str(p.get('current_title', '')).strip()
    current_company      = str(p.get('current_company', '')).strip()
    current_company_size = str(p.get('current_company_size', '')).strip()
    current_industry     = str(p.get('current_industry', '')).strip()

    # Step 3 — skills
    skills: List[SkillEntry] = []
    for item in raw.get('skills', []):
        s_name = str(item['name']).strip().lower()
        s_prof = str(item.get('proficiency', 'beginner'))
        if s_prof not in VALID_PROFICIENCIES:
            s_prof = 'beginner'
        s_end  = max(0, int(item.get('endorsements', 0)))
        s_dur  = max(0, int(item.get('duration_months', 0)))
        # assessment_score filled in Step 7
        skills.append(SkillEntry(
            name=s_name,
            proficiency=s_prof,
            endorsements=s_end,
            duration_months=s_dur,
            assessment_score=-1.0,
        ))

    # Step 4 — career_history
    career_history: List[CareerEntry] = []
    for item in raw.get('career_history', []):
        career_history.append(CareerEntry(
            company        = str(item.get('company', '')).strip(),
            title          = str(item.get('title', '')).strip(),
            start_date     = str(item.get('start_date', '')).strip(),
            end_date       = item.get('end_date'),  # keep None if null
            duration_months= max(0, int(item.get('duration_months', 0))),
            is_current     = bool(item.get('is_current', False)),
            industry       = str(item.get('industry', '')).strip(),
            company_size   = str(item.get('company_size', '')).strip(),
            description    = str(item.get('description', '')).strip(),
        ))

    # Step 5 — education
    education: List[EducationEntry] = []
    for item in raw.get('education', []):
        tier = str(item.get('tier', 'unknown'))
        if tier not in VALID_TIERS:
            tier = 'unknown'
        education.append(EducationEntry(
            institution   = str(item.get('institution', '')).strip(),
            degree        = str(item.get('degree', '')).strip(),
            field_of_study= str(item.get('field_of_study', '')).strip(),
            start_year    = int(item.get('start_year', 0)),
            end_year      = int(item.get('end_year', 0)),
            grade         = item.get('grade'),  # keep None if null/missing
            tier          = tier,
        ))

    # Step 6 — redrob_signals
    sig = raw.get('redrob_signals', {})
    salary_range = sig.get('expected_salary_range_inr_lpa', {})

    redrob_signals = RedrobSignals(
        profile_completeness_score  = float(sig.get('profile_completeness_score', 0.0)),
        signup_date                 = str(sig.get('signup_date', '2000-01-01')),
        last_active_date            = str(sig.get('last_active_date', '2000-01-01')),
        open_to_work_flag           = bool(sig.get('open_to_work_flag', False)),
        profile_views_received_30d  = max(0, int(sig.get('profile_views_received_30d', 0))),
        applications_submitted_30d  = max(0, int(sig.get('applications_submitted_30d', 0))),
        recruiter_response_rate     = float(sig.get('recruiter_response_rate', 0.0)),
        avg_response_time_hours     = float(sig.get('avg_response_time_hours', 999.0)),
        skill_assessment_scores     = {
            str(k).lower(): float(v)
            for k, v in sig.get('skill_assessment_scores', {}).items()
        },
        connection_count            = max(0, int(sig.get('connection_count', 0))),
        endorsements_received       = max(0, int(sig.get('endorsements_received', 0))),
        notice_period_days          = max(0, int(sig.get('notice_period_days', 90))),
        expected_salary_min_lpa     = float(salary_range.get('min', 0.0)),
        expected_salary_max_lpa     = float(salary_range.get('max', 0.0)),
        preferred_work_mode         = str(sig.get('preferred_work_mode', 'flexible')),
        willing_to_relocate         = bool(sig.get('willing_to_relocate', False)),
        github_activity_score       = float(sig.get('github_activity_score', -1.0)),
        search_appearance_30d       = max(0, int(sig.get('search_appearance_30d', 0))),
        saved_by_recruiters_30d     = max(0, int(sig.get('saved_by_recruiters_30d', 0))),
        interview_completion_rate   = float(sig.get('interview_completion_rate', 0.5)),
        offer_acceptance_rate       = float(sig.get('offer_acceptance_rate', -1.0)),
        verified_email              = bool(sig.get('verified_email', False)),
        verified_phone              = bool(sig.get('verified_phone', False)),
        linkedin_connected          = bool(sig.get('linkedin_connected', False)),
    )

    # Step 7 — inject assessment_scores into SkillEntry objects
    for skill in skills:
        if skill.name in redrob_signals.skill_assessment_scores:
            skill.assessment_score = redrob_signals.skill_assessment_scores[skill.name]

    # Step 8 — build full_text for embedding
    parts = [
        current_title,
        headline,
        summary,
        ' '.join(s.name for s in skills),
        ' '.join(f"{c.title} {c.description}" for c in career_history),
        ' '.join(f"{e.degree} {e.field_of_study}" for e in education),
    ]
    full_text = ' '.join(part for part in parts if part.strip())

    # Step 9 — certifications and languages (keep as raw dicts)
    certifications = list(raw.get('certifications', []))
    languages      = list(raw.get('languages', []))

    # Step 10 — Return CandidateProfile
    return CandidateProfile(
        candidate_id        = cid,
        name                = name,
        headline            = headline,
        summary             = summary,
        location            = location,
        country             = country,
        years_of_experience = years_of_experience,
        current_title       = current_title,
        current_company     = current_company,
        current_company_size= current_company_size,
        current_industry    = current_industry,
        skills              = skills,
        career_history      = career_history,
        education           = education,
        certifications      = certifications,
        languages           = languages,
        redrob_signals      = redrob_signals,
        full_text           = full_text,
        embedding           = None,
        is_honeypot         = False,
        honeypot_reasons    = [],
    )
