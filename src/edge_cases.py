"""
edge_cases.py — Module 4
Two responsibilities:
1. Honeypot detection: flags impossible profiles (final_score = 0.0).
2. Edge case adjustments: applies penalties/bonuses to produce final_score.

CRITICAL — ADDITIVE-ONLY RULE:
  Every edge case is a private _ec_* function.
  apply_edge_cases() calls them in FIXED ORDER.
  To add a new edge case: append new _ec_* function + ONE call at END of
  orchestrator, BEFORE the final_score line.
  NEVER modify existing _ec_* functions.
  NEVER reorder existing calls.
"""

from datetime import date
from typing import List
import re

from src.models import CandidateProfile, CandidateScore, JobDescription
from config import (
    CONSULTING_FIRMS, HONEYPOT_TITLES, JD_REQUIRED_SKILLS,
    JD_PREFERRED_NOTICE_DAYS, JD_PREFERRED_LOCATIONS,
    PROFICIENCY_WEIGHTS,
)


# ─────────────────────────────────────────────────────────────────────────────
# MASTER ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def apply_edge_cases(
    candidate: CandidateProfile,
    score: CandidateScore,
    jd: JobDescription,
) -> CandidateScore:
    """
    Run honeypot detection first, then all edge case adjustments.
    Returns CandidateScore with final_score set.

    IMMUTABLE CALL ORDER — do not reorder, do not remove, do not insert in middle.
    New edge cases: append new _ec_* call at the END, before final_score line.
    """
    # ── Phase 1: Honeypot detection ────────────────────────────────────────────
    candidate = _detect_honeypot(candidate)
    if candidate.is_honeypot:
        score.final_score = 0.0
        score.penalties.append(f"honeypot_detected:{';'.join(candidate.honeypot_reasons)}")
        score.penalty_total = score.base_score   # wipe out all score
        return score

    # ── Phase 2: Penalty edge cases ────────────────────────────────────────────
    score = _ec_consulting_only(candidate, score, jd)
    score = _ec_not_open_to_work(candidate, score, jd)
    score = _ec_inactive_long(candidate, score, jd)
    score = _ec_job_hopper(candidate, score, jd)
    score = _ec_keyword_stuffer(candidate, score, jd)
    score = _ec_long_notice_period(candidate, score, jd)
    score = _ec_primary_wrong_domain(candidate, score, jd)
    score = _ec_consulting_only_career(candidate, score, jd)
    score = _ec_pure_research_no_production(candidate, score, jd)
    score = _ec_llm_wrapper_only(candidate, score, jd)
    score = _ec_non_coder_architect(candidate, score, jd)
    score = _ec_cv_speech_robotics_only(candidate, score, jd)
    score = _ec_title_chaser(candidate, score, jd)

    # ── Phase 3: Bonus edge cases ──────────────────────────────────────────────
    score = _ec_preferred_location(candidate, score, jd)
    score = _ec_github_active(candidate, score, jd)
    score = _ec_assessment_excellence(candidate, score, jd)
    score = _ec_internal_promotion(candidate, score, jd)
    score = _ec_open_source_signal(candidate, score, jd)
    score = _ec_short_notice_bonus(candidate, score, jd)
    score = _ec_willing_to_relocate(candidate, score, jd)

    # ── Final score calculation — ALWAYS THE LAST STATEMENT ──────────────────
    # Round to 4dp to match CSV output precision — ensures raw final_score and
    # the written CSV value are identical so non-increasing order tests agree.
    score.final_score = round(max(0.0, min(1.0,
        score.base_score - score.penalty_total + score.bonus_total
    )), 4)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_honeypot(candidate: CandidateProfile) -> CandidateProfile:
    """
    Detect impossibly inconsistent profiles.
    Honeypots are forced to relevance tier 0 in the ground truth.
    Submission with >10% honeypots in top 100 is auto-disqualified.
    Sets candidate.is_honeypot = True and populates candidate.honeypot_reasons.
    """
    reasons = []

    # ── Check 1: Expert proficiency + 0 months duration (impossible) ──────────
    expert_zero_months = [
        s for s in candidate.skills
        if s.proficiency == "expert" and s.duration_months == 0
    ]
    if len(expert_zero_months) >= 3:
        reasons.append(
            f"expert_proficiency_zero_duration:{len(expert_zero_months)}_skills"
        )

    # ── Check 2: Expert + 0 endorsements + 0 duration (triple impossibility) ──
    triple_impossible = [
        s for s in candidate.skills
        if s.proficiency == "expert"
        and s.duration_months == 0
        and s.endorsements == 0
        and s.assessment_score < 0   # no assessment taken either
    ]
    if len(triple_impossible) >= 5:
        reasons.append(
            f"expert_no_evidence_no_assessment:{len(triple_impossible)}_skills"
        )

    # ── Check 3: Claimed years >> sum of career history months ────────────────
    total_career_months = sum(c.duration_months for c in candidate.career_history)
    claimed_months = candidate.years_of_experience * 12
    gap = claimed_months - total_career_months
    if gap > 60:   # claimed 5+ years more than career history shows
        reasons.append(
            f"experience_inflation:{candidate.years_of_experience:.1f}yr_claimed_"
            f"vs_{total_career_months/12:.1f}yr_in_history"
        )

    # ── Check 4: Single role duration exceeding plausible maximum ─────────────
    for career_item in candidate.career_history:
        if career_item.duration_months > 360:   # >30 years in one role
            reasons.append(
                f"impossible_single_role_duration:{career_item.duration_months}mo_"
                f"at_{career_item.company}"
            )

    # ── Check 5: Perfect skill coverage with zero evidence ────────────────────
    # All required JD skills listed as 'expert', all with 0 duration and 0 endorsements
    skill_names = {s.name for s in candidate.skills}
    required_covered = sum(1 for r in JD_REQUIRED_SKILLS if r in skill_names)
    if required_covered >= len(JD_REQUIRED_SKILLS) * 0.8:
        all_expert_no_evidence = all(
            s.duration_months == 0 and s.endorsements == 0
            for s in candidate.skills
            if s.name in JD_REQUIRED_SKILLS
        )
        if all_expert_no_evidence and required_covered >= 5:
            reasons.append("perfect_jd_coverage_zero_evidence")

    if reasons:
        candidate.is_honeypot = True
        candidate.honeypot_reasons = reasons

    return candidate


# ─────────────────────────────────────────────────────────────────────────────
# PENALTY EDGE CASE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _ec_consulting_only(candidate, score, jd):
    """
    JD explicitly states no pure consulting background.
    Only applies if ALL career history is at consulting firms.
    """
    if not candidate.career_history:
        return score
    all_companies = {c.company.lower().strip() for c in candidate.career_history}
    if all_companies and all_companies.issubset(CONSULTING_FIRMS):
        score.penalties.append("consulting_only_career")
        score.penalty_total += 0.15
    return score


def _ec_not_open_to_work(candidate, score, jd):
    """
    Candidate has not marked themselves open to work on Redrob.
    Already downweighted in Stage 1 availability; this adds a further penalty
    at Stage 2 level to ensure they do not appear in top 10.
    """
    if not candidate.redrob_signals.open_to_work_flag:
        score.penalties.append("not_open_to_work")
        score.penalty_total += 0.10
    return score


def _ec_inactive_long(candidate, score, jd):
    """
    Candidate inactive > 90 days. JD notes a perfect-on-paper candidate
    who hasn't logged in for 6 months is 'not actually available'.
    Already partially handled in Stage 1; this adds Stage 2 penalty.
    """
    try:
        last = date.fromisoformat(candidate.redrob_signals.last_active_date)
        days = (date.today() - last).days
    except (ValueError, TypeError):
        days = 365

    if days > 180:
        score.penalties.append(f"inactive_{days}days_severe")
        score.penalty_total += 0.12
    elif days > 90:
        score.penalties.append(f"inactive_{days}days_moderate")
        score.penalty_total += 0.05
    return score


def _ec_job_hopper(candidate, score, jd):
    """
    JD explicitly warns against title-chasers switching every 1.5 years.
    Contract roles (short by nature) are excluded from the calculation.
    """
    contract_keywords = {"contract", "contractual", "c2h", "freelance", "consultant"}
    non_contract = [
        c for c in candidate.career_history
        if not any(kw in c.title.lower() for kw in contract_keywords)
        and c.duration_months > 0
    ]
    if len(non_contract) < 3:
        return score   # not enough data to judge

    avg_tenure = sum(c.duration_months for c in non_contract) / len(non_contract)
    if avg_tenure < 12:
        score.penalties.append(f"job_hopper_avg_{avg_tenure:.0f}mo")
        score.penalty_total += 0.10
    elif avg_tenure < 18:
        score.penalties.append(f"short_tenure_pattern_avg_{avg_tenure:.0f}mo")
        score.penalty_total += 0.04
    return score


def _ec_keyword_stuffer(candidate, score, jd):
    """
    Detect keyword stuffers: career title is irrelevant (e.g. Marketing Manager)
    but skills list contains many JD-required keywords.
    This is the primary trap in the dataset per submission_spec.docx.
    """
    title_lower = candidate.current_title.lower()
    title_is_irrelevant = title_lower in HONEYPOT_TITLES

    skill_names = {s.name for s in candidate.skills}
    jd_skills_present = sum(1 for r in JD_REQUIRED_SKILLS if r in skill_names)

    if title_is_irrelevant and jd_skills_present >= 4:
        score.penalties.append(
            f"keyword_stuffer_title={candidate.current_title}_jd_skills={jd_skills_present}"
        )
        score.penalty_total += 0.25   # severe — this is a confirmed trap candidate
    return score


def _ec_long_notice_period(candidate, score, jd):
    """
    JD notes preference for sub-30-day notice.
    Long notice doesn't disqualify but reduces ranking among equals.
    """
    days = candidate.redrob_signals.notice_period_days
    if days > 90:
        score.penalties.append(f"long_notice_{days}days")
        score.penalty_total += 0.04
    elif days > 60:
        score.penalties.append(f"elevated_notice_{days}days")
        score.penalty_total += 0.02
    return score


def _ec_primary_wrong_domain(candidate, score, jd):
    """
    JD explicitly says no candidates whose primary focus is CV, speech, robotics, audio.
    Detect via skill set and career description.
    """
    wrong_domain_skills = {
        "computer vision", "object detection", "image segmentation",
        "speech recognition", "text-to-speech", "tts", "audio processing",
        "robotics", "ros", "autonomous vehicles", "image classification",
        "ocr", "optical character recognition",
    }
    candidate_skills = {s.name for s in candidate.skills}
    wrong_matches = candidate_skills.intersection(wrong_domain_skills)

    # Only penalise if CV/speech skills outnumber IR/retrieval skills
    ir_skills = {
        "embeddings", "retrieval", "ranking", "vector database", "faiss",
        "pinecone", "weaviate", "qdrant", "semantic search", "elasticsearch",
    }
    ir_matches = candidate_skills.intersection(ir_skills)

    if len(wrong_matches) > len(ir_matches) + 2:
        score.penalties.append(
            f"primary_wrong_domain:{','.join(list(wrong_matches)[:3])}"
        )
        score.penalty_total += 0.08
    return score


# ─────────────────────────────────────────────────────────────────────────────
# PENALTY EDGE CASE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _ec_consulting_only_career(candidate, score, jd):
    if not candidate.career_history:
        return score
    all_companies = {c.company.lower().strip() for c in candidate.career_history}
    is_all_consulting = all_companies and all_companies.issubset(CONSULTING_FIRMS)
    has_product = not is_all_consulting # simplified product company check
    if is_all_consulting and not has_product:
        score.penalties.append("consulting_only_career_hard")
        score.penalty_total += score.base_score * 0.95
    return score


def _ec_pure_research_no_production(candidate, score, jd):
    if not candidate.career_history:
        return score
    academic_keywords = {"university", "research lab", "institute", "postdoc", "phd", "academic"}
    production_signals = ["production", "deployed", "shipped", "real users", "scale", "serving"]
    all_descriptions = " ".join(c.description.lower() for c in candidate.career_history)
    all_academic = all(any(kw in c.company.lower() or kw in c.title.lower() for kw in academic_keywords) for c in candidate.career_history)
    has_prod = any(sig in all_descriptions for sig in production_signals)
    if all_academic and not has_prod:
        score.penalties.append("pure_research_no_production")
        score.penalty_total += score.base_score * 0.90
    return score


def _ec_llm_wrapper_only(candidate, score, jd):
    wrapper_keywords = {"langchain", "llamaindex", "openai api"}
    has_wrapper = any(s.name.lower() in wrapper_keywords for s in candidate.skills)
    max_wrapper_duration = max([s.duration_months for s in candidate.skills if s.name.lower() in wrapper_keywords], default=0)
    has_pre_llm = any(s.duration_months > 12 and s.name.lower() not in wrapper_keywords for s in candidate.skills if s.proficiency in ["advanced", "expert"])
    if has_wrapper and max_wrapper_duration < 12 and not has_pre_llm:
        score.penalties.append("llm_wrapper_only_under_12m")
        score.penalty_total += score.base_score * 0.85
    return score


def _ec_non_coder_architect(candidate, score, jd):
    title = candidate.current_title.lower()
    senior_titles = {"vp", "director", "principal architect"}
    is_senior = any(st in title for st in senior_titles)
    github_score = candidate.redrob_signals.github_activity_score
    if is_senior and github_score <= 0:
        score.penalties.append("non_coder_architect")
        score.penalty_total += score.base_score * 0.80
    return score


def _ec_cv_speech_robotics_only(candidate, score, jd):
    wrong_domain_skills = {"computer vision", "object detection", "image segmentation", "speech recognition", "tts", "robotics", "ros"}
    ir_skills = {"embeddings", "retrieval", "ranking", "vector database", "faiss", "pinecone", "weaviate", "qdrant", "semantic search", "elasticsearch", "nlp"}
    candidate_skills = {s.name for s in candidate.skills}
    wrong_matches = candidate_skills.intersection(wrong_domain_skills)
    ir_matches = candidate_skills.intersection(ir_skills)
    if len(wrong_matches) > 0 and len(ir_matches) == 0:
        score.penalties.append("cv_speech_robotics_only")
        score.penalty_total += score.base_score * 0.85
    return score


def _ec_title_chaser(candidate, score, jd):
    if len(candidate.career_history) >= 3:
        # Sort by start date to find span
        sorted_history = sorted(candidate.career_history, key=lambda c: c.start_date)
        try:
            from datetime import date
            start = date.fromisoformat(sorted_history[0].start_date)
            end = date.fromisoformat(sorted_history[-1].end_date) if sorted_history[-1].end_date else date.today()
            span_years = (end - start).days / 365.0
            max_tenure = max(c.duration_months for c in candidate.career_history)
            if span_years < 4.5 and max_tenure <= 18:
                score.penalties.append("title_chaser_pattern")
                score.penalty_total += score.base_score * 0.30  # Soft penalty (multiplier 0.7x)
        except:
            pass
    return score

# ─────────────────────────────────────────────────────────────────────────────
# BONUS EDGE CASE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _ec_preferred_location(candidate, score, jd):
    """Bonus if already in a JD-preferred city (Pune, Noida, Hyderabad, etc.)."""
    location_lower = candidate.location.lower()
    if any(city in location_lower for city in JD_PREFERRED_LOCATIONS):
        score.bonuses.append(f"preferred_location:{candidate.location}")
        score.bonus_total += 0.03
    return score


def _ec_github_active(candidate, score, jd):
    """Bonus for high GitHub activity — signals active engineering work."""
    gh = candidate.redrob_signals.github_activity_score
    if gh >= 70:
        score.bonuses.append(f"strong_github_activity:{gh:.0f}")
        score.bonus_total += 0.06
    elif gh >= 40:
        score.bonuses.append(f"moderate_github_activity:{gh:.0f}")
        score.bonus_total += 0.03
    return score


def _ec_assessment_excellence(candidate, score, jd):
    """
    Bonus if any JD-relevant skill has an assessment score >= 75.
    Assessment is an objective third-party measure — highest-trust signal.
    """
    assessments = candidate.redrob_signals.skill_assessment_scores
    for skill_name, ass_score in assessments.items():
        if ass_score >= 75:
            for req in JD_REQUIRED_SKILLS:
                if req in skill_name or skill_name in req:
                    score.bonuses.append(
                        f"high_assessment_{skill_name}:{ass_score:.0f}"
                    )
                    score.bonus_total += 0.06
                    return score   # one bonus per candidate
    return score


def _ec_internal_promotion(candidate, score, jd):
    """
    Bonus for candidates who were promoted internally (multiple roles same company).
    Signals reliability, growth, and non-job-hopping behaviour.
    """
    company_counts = {}
    for c in candidate.career_history:
        key = c.company.lower().strip()
        company_counts[key] = company_counts.get(key, 0) + 1

    if any(count >= 2 for count in company_counts.values()):
        score.bonuses.append("internal_promotion_detected")
        score.bonus_total += 0.04
    return score


def _ec_open_source_signal(candidate, score, jd):
    """
    JD lists open-source contributions as preferred.
    High GitHub score + relevant skills = strong signal.
    Only applies when GitHub is linked (score != -1).
    """
    gh = candidate.redrob_signals.github_activity_score
    if gh < 0:
        return score   # no GitHub linked

    oss_keywords = ["open source", "github", "contributor", "maintainer", "oss"]
    all_text = candidate.full_text.lower()
    has_oss_mention = any(kw in all_text for kw in oss_keywords)

    if gh >= 50 and has_oss_mention:
        score.bonuses.append(f"open_source_contributor:github={gh:.0f}")
        score.bonus_total += 0.04
    return score


def _ec_short_notice_bonus(candidate, score, jd):
    """Bonus for notice period at or below JD preference (30 days)."""
    days = candidate.redrob_signals.notice_period_days
    if days <= JD_PREFERRED_NOTICE_DAYS:
        score.bonuses.append(f"preferred_notice_{days}days")
        score.bonus_total += 0.03
    return score


def _ec_willing_to_relocate(candidate, score, jd):
    """Small bonus if outside preferred cities but willing to relocate."""
    location_lower = candidate.location.lower()
    already_preferred = any(city in location_lower for city in JD_PREFERRED_LOCATIONS)
    if not already_preferred and candidate.redrob_signals.willing_to_relocate:
        score.bonuses.append("willing_to_relocate")
        score.bonus_total += 0.02
    return score
