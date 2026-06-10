"""
scoring_engine.py — Module 3
Two-stage scoring pipeline.
Stage 1: Fast structured scoring on ALL 100,000 candidates (no embeddings).
Stage 2: Full 6-component scoring on top 5,000 (uses embeddings).
No edge case logic, no penalties, no bonuses — those live in Module 4.
"""

from datetime import date, datetime
from typing import List, Optional, Tuple
import numpy as np

from src.models import CandidateProfile, JobDescription, CandidateScore, SkillEntry
from src.nlp_engine import cosine_similarity
from config import (
    W1_TITLE, W1_SKILLS, W1_AVAILABILITY, W1_DOMAIN,
    W2_SEMANTIC, W2_SKILL_QUALITY, W2_CAREER_FIT,
    W2_EXPERIENCE, W2_BEHAVIORAL, W2_EDUCATION,
    PROFICIENCY_WEIGHTS, EDUCATION_TIER_SCORES,
    JD_REQUIRED_SKILLS, JD_PREFERRED_SKILLS,
    JD_EXPERIENCE_MIN, JD_EXPERIENCE_MAX,
    SKILL_SYNONYMS, SKILL_FAMILY_GROUPS,
    AI_ML_TITLES, HONEYPOT_TITLES, CONSULTING_FIRMS,
)


# ═══════════════════════════════════════════════════
# STAGE 1 — FAST FILTER (handles all 100,000)
# ═══════════════════════════════════════════════════

def compute_stage1_score(candidate: CandidateProfile, jd: JobDescription) -> float:
    """
    Fast structured score. No embeddings. Returns float 0.0-1.0.
    Called on ALL 100,000 candidates. Must be extremely fast.
    """
    title_score  = _s1_title_relevance(candidate) * W1_TITLE
    skills_score = _s1_skill_count(candidate, jd)  * W1_SKILLS
    avail_score  = _s1_availability(candidate)      * W1_AVAILABILITY
    domain_score = _s1_domain_fit(candidate)        * W1_DOMAIN

    raw = title_score + skills_score + avail_score + domain_score
    return max(0.0, min(1.0, raw))


def _s1_title_relevance(candidate: CandidateProfile) -> float:
    """
    Is this person's current title relevant to an AI/ML engineering role?
    The single most discriminating fast signal in the dataset.
    """
    title_lower = candidate.current_title.lower().strip()

    # Step 1: Hard disqualify — honeypot titles
    if title_lower in HONEYPOT_TITLES:
        return 0.0

    # Step 2: Strong positive — AI/ML titles
    for ai_title in AI_ML_TITLES:
        if ai_title in title_lower:
            return 1.0

    # Step 3: Moderate positive — technical but not AI-specific
    technical_general = {
        "software engineer", "backend engineer", "data engineer",
        "full stack", "platform engineer", "devops engineer",
        "cloud engineer", "site reliability", "sre", "architect",
    }
    for t in technical_general:
        if t in title_lower:
            return 0.55

    # Step 4: Weak positive — adjacent technical
    weak_technical = {
        "frontend engineer", "qa engineer", "mobile engineer",
        "android engineer", "ios engineer", "java developer",
        ".net developer", "php developer",
    }
    for t in weak_technical:
        if t in title_lower:
            return 0.25

    # Step 5: Unknown / not matched
    return 0.15


def _s1_skill_count(candidate: CandidateProfile, jd: JobDescription) -> float:
    """
    How many of the JD required skills does the candidate have?
    Fast — only checks presence, not quality. Quality checked in Stage 2.
    """
    candidate_skills = {s.name for s in candidate.skills}  # already lowercased

    matches = 0.0
    for required_skill in JD_REQUIRED_SKILLS:
        if required_skill in candidate_skills:
            matches += 1.0
            continue
        # Check synonyms
        synonym_matched = False
        for synonym in SKILL_SYNONYMS.get(required_skill, []):
            if synonym in candidate_skills:
                matches += 0.8  # synonym = 80% of direct match
                synonym_matched = True
                break
        if synonym_matched:
            continue
        # Check family groups (partial credit)
        for family_name, family_members in SKILL_FAMILY_GROUPS.items():
            if required_skill in family_members:
                for member in family_members:
                    if member in candidate_skills and member != required_skill:
                        matches += 0.6  # same family = 60% credit
                        break
                break

    ratio = matches / max(len(JD_REQUIRED_SKILLS), 1)
    return min(1.0, ratio)


def _s1_availability(candidate: CandidateProfile) -> float:
    """
    Is this candidate actually available and active?
    A perfect-on-paper candidate who is inactive for 6 months is not useful.
    """
    sig = candidate.redrob_signals

    # open_to_work is the primary signal
    if not sig.open_to_work_flag:
        open_weight = 0.3  # not open = severe downweight
    else:
        open_weight = 1.0

    # Last active recency
    try:
        last = date.fromisoformat(sig.last_active_date)
        days_inactive = (date.today() - last).days
    except (ValueError, TypeError):
        days_inactive = 365  # default to inactive if date unreadable

    if days_inactive <= 14:
        recency = 1.0
    elif days_inactive <= 30:
        recency = 0.9
    elif days_inactive <= 60:
        recency = 0.75
    elif days_inactive <= 90:
        recency = 0.60
    elif days_inactive <= 180:
        recency = 0.40
    else:
        recency = 0.15  # >6 months = nearly unavailable

    availability = (open_weight * 0.60) + (recency * 0.40)
    return max(0.0, min(1.0, availability))


def _s1_domain_fit(candidate: CandidateProfile) -> float:
    """Is the candidate in a tech/AI-adjacent domain at all?"""
    tech_industries = {
        "technology", "software", "information technology", "internet",
        "computer software", "artificial intelligence", "machine learning",
        "data analytics", "cloud computing", "saas", "fintech", "edtech",
        "ecommerce", "e-commerce", "healthtech", "deeptech", "startup",
    }

    industry_lower = candidate.current_industry.lower()
    for ti in tech_industries:
        if ti in industry_lower:
            return 1.0

    # Check career history industries — only recent 3 roles
    for career_item in candidate.career_history[:3]:
        for ti in tech_industries:
            if ti in career_item.industry.lower():
                return 0.75

    return 0.25  # non-tech but not disqualified yet


# ═══════════════════════════════════════════════════
# STAGE 2 — FULL SCORING (handles top 5,000)
# ═══════════════════════════════════════════════════

def compute_base_score(
    candidate: CandidateProfile,
    jd: JobDescription,
) -> CandidateScore:
    """
    Full multi-factor scoring. Requires candidate.embedding and jd.embedding to be set.
    Returns CandidateScore with all 6 component scores and base_score populated.
    base_score is in [0.0, 1.0]. Penalties and bonuses applied later in Module 4.
    """
    assert candidate.embedding is not None, \
        f"embed_candidates_batch must run before compute_base_score (id={candidate.candidate_id})"
    assert jd.embedding is not None, \
        "embed_jd must run before compute_base_score"

    semantic      = _s2_semantic(candidate, jd)
    skill_quality = _s2_skill_quality(candidate, jd)
    career_fit    = _s2_career_fit(candidate, jd)
    experience    = _s2_experience(candidate, jd)
    behavioral    = _s2_behavioral(candidate)
    education     = _s2_education(candidate)

    base = (
        semantic      * W2_SEMANTIC      +
        skill_quality * W2_SKILL_QUALITY +
        career_fit    * W2_CAREER_FIT    +
        experience    * W2_EXPERIENCE    +
        behavioral    * W2_BEHAVIORAL    +
        education     * W2_EDUCATION
    )

    return CandidateScore(
        candidate_id        = candidate.candidate_id,
        semantic_score      = semantic,
        skill_quality_score = skill_quality,
        career_fit_score    = career_fit,
        experience_score    = experience,
        behavioral_score    = behavioral,
        education_score     = education,
        base_score          = max(0.0, min(1.0, base)),
    )


def _s2_semantic(candidate: CandidateProfile, jd: JobDescription) -> float:
    """Semantic similarity between full candidate profile and the JD."""
    sim = cosine_similarity(candidate.embedding, jd.embedding)
    return max(0.0, min(1.0, float(sim)))


def _s2_skill_quality(candidate: CandidateProfile, jd: JobDescription) -> float:
    """
    Quality-weighted skill score. Goes beyond presence to measure
    depth via proficiency + duration + endorsements + assessment scores.
    This is the most important Stage 2 signal (W2_SKILL_QUALITY = 0.30).
    """
    candidate_skills_map = {s.name: s for s in candidate.skills}

    required_scores = []
    for req in JD_REQUIRED_SKILLS:
        skill = _find_skill(req, candidate_skills_map)
        if skill is None:
            required_scores.append(0.0)  # missing = 0
            continue

        # Base from proficiency
        base = PROFICIENCY_WEIGHTS[skill.proficiency]

        # Duration bonus: +0.2 max at 48+ months (4 years)
        duration_bonus = min(0.2, skill.duration_months / 240.0)

        # Endorsement bonus: +0.1 max at 100+ endorsements
        endorsement_bonus = min(0.1, skill.endorsements / 1000.0)

        if skill.assessment_score >= 0:
            # Assessment score overrides proficiency as the primary signal
            # because it's an objective third-party measure
            assessment_component = skill.assessment_score / 100.0
            quality = (assessment_component * 0.6) + (base * 0.3) + (duration_bonus * 0.1)
        else:
            quality = base + duration_bonus + endorsement_bonus

        required_scores.append(min(1.0, quality))

    # Preferred skills add a smaller bonus
    preferred_scores = []
    for pref in JD_PREFERRED_SKILLS:
        skill = _find_skill(pref, candidate_skills_map)
        if skill:
            base = PROFICIENCY_WEIGHTS[skill.proficiency]
            preferred_scores.append(base * 0.5)  # preferred worth 50% of required

    if not required_scores and not preferred_scores:
        return 0.0

    required_avg = sum(required_scores) / max(len(required_scores), 1)
    preferred_avg = sum(preferred_scores) / max(len(preferred_scores), 1) if preferred_scores else 0.0

    final = (required_avg * 0.80) + (preferred_avg * 0.20)
    return max(0.0, min(1.0, final))


def _find_skill(skill_name: str, skills_map: dict) -> Optional[SkillEntry]:
    """
    Helper used by _s2_skill_quality.
    Looks up a skill by name, including synonyms and family groups.
    Returns the matching SkillEntry or None.
    """
    # Direct match
    if skill_name in skills_map:
        return skills_map[skill_name]

    # Synonym match
    for synonym in SKILL_SYNONYMS.get(skill_name, []):
        if synonym in skills_map:
            return skills_map[synonym]

    # Family group match (partial — returns best member found)
    for family_members in SKILL_FAMILY_GROUPS.values():
        if skill_name in family_members:
            best = None
            best_weight = -1.0
            for member in family_members:
                if member in skills_map:
                    s = skills_map[member]
                    w = PROFICIENCY_WEIGHTS[s.proficiency] + s.duration_months / 100
                    if w > best_weight:
                        best, best_weight = s, w
            return best  # may still be None if no family member in candidate

    return None


def _s2_career_fit(candidate: CandidateProfile, jd: JobDescription) -> float:
    """
    How well does the candidate's career history match the JD?
    Checks: title trajectory, production ML experience, company type.
    """
    score = 0.0

    # Component 1: Title trajectory (30% of career_fit)
    ai_title_count = 0
    for career_item in candidate.career_history:
        title_lower = career_item.title.lower()
        for ai_title in AI_ML_TITLES:
            if ai_title in title_lower:
                ai_title_count += 1
                break

    title_trajectory = min(1.0, ai_title_count / max(len(candidate.career_history), 1))
    score += title_trajectory * 0.30

    # Component 2: Production ML signals in career descriptions (40% of career_fit)
    production_signals = [
        "production", "deployed", "shipped", "real users", "scale",
        "latency", "throughput", "a/b test", "serving", "inference",
        "ranking", "retrieval", "search", "recommendation", "embedding",
        "vector", "faiss", "pinecone", "qdrant", "elasticsearch",
    ]
    all_descriptions = " ".join(c.description.lower() for c in candidate.career_history)
    signal_count = sum(1 for sig in production_signals if sig in all_descriptions)
    production_score = min(1.0, signal_count / 8.0)  # full score at 8+ signals
    score += production_score * 0.40

    # Component 3: Company type — product vs consulting (30% of career_fit)
    consulting_count = sum(
        1 for c in candidate.career_history
        if c.company.lower() in CONSULTING_FIRMS
    )
    total_roles = max(len(candidate.career_history), 1)
    consulting_ratio = consulting_count / total_roles

    if consulting_ratio >= 1.0:
        company_score = 0.0   # all consulting
    elif consulting_ratio > 0.5:
        company_score = 0.3   # majority consulting
    elif consulting_ratio > 0.0:
        company_score = 0.7   # mixed
    else:
        company_score = 1.0   # no consulting at all

    score += company_score * 0.30

    return max(0.0, min(1.0, score))


def _s2_experience(candidate: CandidateProfile, jd: JobDescription) -> float:
    """Years of experience relative to JD requirements (5-9 years)."""
    years = candidate.years_of_experience

    if JD_EXPERIENCE_MIN <= years <= JD_EXPERIENCE_MAX:
        return 1.0  # ideal range

    if years < JD_EXPERIENCE_MIN:
        deficit = JD_EXPERIENCE_MIN - years
        return max(0.0, 1.0 - (deficit * 0.18))  # -18% per year short

    # Overqualified (> 9 years)
    excess = years - JD_EXPERIENCE_MAX
    return max(0.4, 1.0 - (excess * 0.06))  # soft penalty, never below 0.4


def _s2_behavioral(candidate: CandidateProfile) -> float:
    """
    Engagement and availability signals from the Redrob platform.
    These are multipliers on core fit — a highly engaged candidate with
    wrong skills still ranks below a less engaged candidate with right skills.
    """
    sig = candidate.redrob_signals

    # Response rate: 30% weight
    response_score = sig.recruiter_response_rate  # already 0.0-1.0

    # Profile completeness: 15%
    completeness = sig.profile_completeness_score / 100.0

    # GitHub activity: 25%
    if sig.github_activity_score < 0:
        github = 0.35  # neutral if no GitHub (don't penalise absence)
    else:
        github = sig.github_activity_score / 100.0

    # Interview completion: 15%
    interview = sig.interview_completion_rate

    # Market demand: 15%
    demand_raw = (sig.saved_by_recruiters_30d * 10 + sig.search_appearance_30d) / 200.0
    demand = min(1.0, demand_raw)

    behavioral = (
        response_score * 0.30 +
        completeness   * 0.15 +
        github         * 0.25 +
        interview      * 0.15 +
        demand         * 0.15
    )
    return max(0.0, min(1.0, behavioral))


def _s2_education(candidate: CandidateProfile) -> float:
    """
    Education tier and field relevance.
    Least important signal at W2_EDUCATION = 0.05.
    For senior ML engineers, tier matters less than what they've built.
    """
    if not candidate.education:
        return 0.40  # neutral default, no education listed

    # Best tier from any degree
    best_tier = max(
        EDUCATION_TIER_SCORES.get(e.tier, 0.40)
        for e in candidate.education
    )

    # STEM field bonus
    stem_keywords = {
        "computer science", "software", "data science", "mathematics",
        "statistics", "electrical", "electronics", "information technology",
        "computer engineering", "artificial intelligence", "machine learning",
    }
    all_fields = " ".join(e.field_of_study.lower() for e in candidate.education)
    stem_bonus = 0.10 if any(k in all_fields for k in stem_keywords) else 0.0

    return min(1.0, best_tier + stem_bonus)
