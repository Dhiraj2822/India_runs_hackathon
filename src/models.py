"""
models.py — All dataclasses for the Redrob Candidate Ranking System.
Build this file first. All modules import from here.
DO NOT add fields not listed here without updating all downstream modules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Sub-objects that appear inside CandidateProfile
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SkillEntry:
    """One skill from the candidate's skills array."""
    name: str                       # lowercased, stripped
    proficiency: str                # "beginner"|"intermediate"|"advanced"|"expert"
    endorsements: int               # integer >= 0
    duration_months: int            # months used, integer >= 0
    assessment_score: float = -1.0  # -1.0 = not assessed. 0-100 if assessed.


@dataclass
class CareerEntry:
    """One role from career_history array."""
    company: str
    title: str
    start_date: str                 # ISO date string "YYYY-MM-DD"
    end_date: Optional[str]         # None if current role
    duration_months: int
    is_current: bool
    industry: str
    company_size: str               # enum: "1-10"|"11-50"|...|"10001+"
    description: str                # free text role description


@dataclass
class EducationEntry:
    """One education record."""
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str]            # GPA / percentage / class — may be None
    tier: str                       # "tier_1"|"tier_2"|"tier_3"|"tier_4"|"unknown"


@dataclass
class RedrobSignals:
    """All 23 platform behavioral signals."""
    profile_completeness_score: float   # 0-100
    signup_date: str
    last_active_date: str               # ISO date "YYYY-MM-DD"
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float      # 0.0-1.0
    avg_response_time_hours: float      # hours
    skill_assessment_scores: Dict[str, float]  # {skill_name: 0-100}
    connection_count: int
    endorsements_received: int
    notice_period_days: int             # 0-180
    expected_salary_min_lpa: float
    expected_salary_max_lpa: float
    preferred_work_mode: str            # "remote"|"hybrid"|"onsite"|"flexible"
    willing_to_relocate: bool
    github_activity_score: float        # -1 = no GitHub linked; 0-100 otherwise
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float    # 0.0-1.0
    offer_acceptance_rate: float        # -1 = no history; 0.0-1.0 otherwise
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


# ─────────────────────────────────────────────────────────────────────────────
# Top-level objects
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateProfile:
    """Complete candidate profile. Populated by data_loader.py."""
    candidate_id: str
    name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str
    skills: List[SkillEntry]
    career_history: List[CareerEntry]
    education: List[EducationEntry]
    certifications: List[dict]          # raw dicts: {name, issuer, year}
    languages: List[dict]               # raw dicts: {language, proficiency}
    redrob_signals: RedrobSignals
    full_text: str                      # concatenated profile text for embedding
    embedding: Optional[np.ndarray] = None  # set by nlp_engine.py
    is_honeypot: bool = False           # set by edge_cases.py
    honeypot_reasons: List[str] = field(default_factory=list)


@dataclass
class JobDescription:
    """Parsed job description."""
    raw_text: str
    required_skills: List[str]          # lowercase skill names
    preferred_skills: List[str]
    experience_min: float
    experience_max: float
    seniority_level: str                # "senior"
    domain: str                         # "ai_ml"
    preferred_locations: List[str]      # lowercase city names
    disqualified_titles: List[str]      # lowercase — from JD explicit text
    disqualified_companies: List[str]   # lowercase — consulting firms from JD
    embedding: Optional[np.ndarray] = None  # set by nlp_engine.py


@dataclass
class CandidateScore:
    """Full score breakdown for one candidate against one JD."""
    candidate_id: str

    # Stage 1 fast score (0.0-1.0)
    stage1_score: float = 0.0

    # Stage 2 component scores (each 0.0-1.0)
    semantic_score: float = 0.0
    skill_quality_score: float = 0.0
    career_fit_score: float = 0.0
    experience_score: float = 0.0
    behavioral_score: float = 0.0
    education_score: float = 0.0

    # Weighted base score before edge case adjustments (0.0-1.0)
    base_score: float = 0.0

    # Edge case adjustments (accumulated by edge_cases.py)
    penalties: List[str] = field(default_factory=list)
    penalty_total: float = 0.0
    bonuses: List[str] = field(default_factory=list)
    bonus_total: float = 0.0

    # Final score and rank (set by ranker.py)
    final_score: float = 0.0     # ALWAYS in [0.0, 1.0]
    rank: int = 0

    # Output fields (set by reasoning.py)
    reasoning: str = ""
