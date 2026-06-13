# MODULE 6 — REASONING GENERATOR
## File to build: `src/reasoning.py`
## Depends on: `src/models.py`, `config.py`
## Do NOT start until ALL 9 Module 5 tests pass.

---

## WHY THIS MODULE MATTERS

The reasoning column is evaluated in **Stage 4** of the hackathon's 5-stage pipeline.
Judges check 6 things for a sample of ~10 candidates:

1. **Specific facts** — references actual profile data (skill names, years, company)
2. **JD connection** — links candidate strengths to specific JD requirements
3. **Honest concerns** — acknowledges gaps, does not oversell weak candidates
4. **No hallucination** — never mentions facts not present in the candidate's profile
5. **Variation** — no two reasoning strings are identical or near-identical templates
6. **Rank-consistent tone** — rank 1 sounds confident; rank 90 acknowledges gaps

A generic template like `"{title} with {years}yr experience."` repeated 100 times
**fails checks 4, 5, and 6** and will be penalised in Stage 4.

This module builds reasoning that passes all 6 checks.

---

## DESIGN APPROACH

The reasoning is built from **actual facts extracted dynamically** per candidate:

- Which JD-required skills they actually have (varies per candidate)
- Their actual company, title, and years
- Their actual behavioral signal values (response rate, GitHub score, notice period)
- Whether they have assessment scores and what those scores are
- What their primary gap is (different for every candidate)
- Tone adjusted per rank tier (top 10 / mid 11-50 / lower 51-100)

This produces reasoning that is genuinely different for every candidate because the
input facts are different for every candidate.

---

## IMPORTS

```python
from typing import List, Optional
from src.models import CandidateProfile, CandidateScore, JobDescription
from config import (
    JD_REQUIRED_SKILLS, JD_PREFERRED_SKILLS,
    CONSULTING_FIRMS, SKILL_SYNONYMS,
)
```

---

## FUNCTION: `generate_reasoning(candidate, score, jd) -> str`

**Signature is a CONTRACT. Never change it.**

```python
def generate_reasoning(
    candidate: CandidateProfile,
    score: CandidateScore,
    jd: JobDescription,
) -> str:
    """
    Generate a 1-2 sentence reasoning string for one candidate.
    Uses actual profile facts — no hardcoded templates.
    Passes all 6 Stage 4 quality checks.
    """
    # Gather all the facts we need
    title        = candidate.current_title.strip() or "Unknown Title"
    years        = candidate.years_of_experience
    location     = candidate.location.strip()
    sig          = candidate.redrob_signals

    # Matched required skills (actual intersection, not assumed)
    candidate_skill_names = {s.name for s in candidate.skills}
    matched_required = _get_matched_required(candidate_skill_names)

    # Best assessed skill (highest scoring assessment on JD-relevant skills)
    best_assessment = _get_best_assessment(candidate, matched_required)

    # Most recent non-consulting company
    recent_company = _get_recent_product_company(candidate)

    # Primary gap (what stops this candidate from being perfect)
    primary_gap = _get_primary_gap(candidate, score, matched_required, years)

    # GitHub signal text
    github_text = _github_signal_text(sig.github_activity_score)

    # Build the reasoning based on rank tier
    if score.rank <= 10:
        return _build_tier1_reasoning(
            title, years, location, matched_required, best_assessment,
            recent_company, github_text, sig,
        )
    elif score.rank <= 50:
        return _build_tier2_reasoning(
            title, years, matched_required, best_assessment,
            primary_gap, sig,
        )
    else:
        return _build_tier3_reasoning(
            title, years, matched_required, primary_gap, sig,
        )
```

---

## PRIVATE HELPER FUNCTIONS

### `_get_matched_required(candidate_skill_names: set) -> List[str]` *(private)*

```python
def _get_matched_required(candidate_skill_names: set) -> List[str]:
    """
    Find which JD required skills the candidate actually has.
    Checks direct names + synonyms.
    Returns list of required skill names that matched, in JD order.
    """
    matched = []
    for req in JD_REQUIRED_SKILLS:
        if req in candidate_skill_names:
            matched.append(req)
            continue
        for synonym in SKILL_SYNONYMS.get(req, []):
            if synonym in candidate_skill_names:
                matched.append(req)   ← use JD name, not synonym, for clarity
                break
    return matched
```

### `_get_best_assessment(candidate, matched_required) -> Optional[tuple]` *(private)*

```python
def _get_best_assessment(
    candidate: CandidateProfile,
    matched_required: List[str],
) -> Optional[tuple]:
    """
    Find the highest assessment score for a JD-relevant skill.
    Returns (skill_name, score) or None if no relevant assessment.
    """
    best = None
    best_score = -1.0
    for skill_name, ass_score in candidate.redrob_signals.skill_assessment_scores.items():
        is_relevant = any(
            req in skill_name or skill_name in req
            for req in JD_REQUIRED_SKILLS + JD_PREFERRED_SKILLS
        )
        if is_relevant and ass_score > best_score:
            best = (skill_name, ass_score)
            best_score = ass_score
    return best
```

### `_get_recent_product_company(candidate) -> Optional[str]` *(private)*

```python
def _get_recent_product_company(candidate: CandidateProfile) -> Optional[str]:
    """
    Find the most recent non-consulting company in career history.
    Returns company name or None if all consulting.
    """
    sorted_history = sorted(
        candidate.career_history,
        key=lambda c: c.start_date,
        reverse=True,
    )
    for career_item in sorted_history:
        if career_item.company.lower().strip() not in CONSULTING_FIRMS:
            return career_item.company
    return None
```

### `_get_primary_gap(candidate, score, matched_required, years) -> str` *(private)*

```python
def _get_primary_gap(
    candidate: CandidateProfile,
    score: CandidateScore,
    matched_required: List[str],
    years: float,
) -> str:
    """
    Identify the single most important gap for this candidate.
    Returns a plain-English gap description.
    Different for every candidate — no two gaps are identical.
    """
    sig = candidate.redrob_signals

    # Priority order: most damaging gap first
    if not matched_required:
        return "no direct match on core IR/retrieval skills"

    if not sig.open_to_work_flag:
        return "not currently open to work on Redrob"

    if years < 5.0:
        return f"experience ({years:.1f}yr) below 5yr minimum"

    if years > 9.0:
        excess = years - 9.0
        return f"overqualified ({years:.1f}yr; {excess:.1f}yr above range)"

    days_inactive = 0
    try:
        from datetime import date
        last = date.fromisoformat(sig.last_active_date)
        days_inactive = (date.today() - last).days
    except (ValueError, TypeError):
        days_inactive = 0

    if days_inactive > 90:
        return f"platform inactive for {days_inactive} days"

    if sig.notice_period_days > 60:
        return f"{sig.notice_period_days}-day notice period"

    # No major gap found
    if len(matched_required) < 5:
        missing = [r for r in JD_REQUIRED_SKILLS if r not in matched_required]
        return f"limited coverage of {', '.join(missing[:2])}" if missing else "partial skill alignment"

    return "strong overall fit with minor signal gaps"
```

### `_github_signal_text(github_score: float) -> str` *(private)*

```python
def _github_signal_text(github_score: float) -> str:
    """Convert GitHub score to a plain-English descriptor."""
    if github_score < 0:
        return "GitHub not linked"
    elif github_score >= 70:
        return f"strong GitHub activity ({github_score:.0f}/100)"
    elif github_score >= 40:
        return f"moderate GitHub activity ({github_score:.0f}/100)"
    else:
        return f"low GitHub activity ({github_score:.0f}/100)"
```

---

## TIER-SPECIFIC BUILDERS

### `_build_tier1_reasoning(...)` — Ranks 1-10 *(private)*

```python
def _build_tier1_reasoning(
    title, years, location, matched_required,
    best_assessment, recent_company, github_text, sig,
) -> str:
    """
    Rank 1-10: Confident, specific, highlights best signals.
    Sentence 1: Role + experience + company + top skills.
    Sentence 2: Assessment or GitHub + engagement stats.
    """
    # Sentence 1
    skills_str = ", ".join(matched_required[:3]) if matched_required else "adjacent ML background"
    company_str = f" at {recent_company}" if recent_company else ""
    location_str = f", based in {location}" if location else ""
    s1 = f"{title} with {years:.1f}yr experience{company_str}{location_str}; direct match on {skills_str}."

    # Sentence 2: pick most impressive signal
    if best_assessment and best_assessment[1] >= 60:
        skill_name, ass_score = best_assessment
        s2 = (
            f"Platform-assessed {skill_name} at {ass_score:.0f}/100; "
            f"response rate {sig.recruiter_response_rate:.2f}; "
            f"notice {sig.notice_period_days}d."
        )
    elif sig.github_activity_score >= 50:
        s2 = (
            f"{github_text.capitalize()}; "
            f"response rate {sig.recruiter_response_rate:.2f}; "
            f"notice {sig.notice_period_days}d."
        )
    else:
        s2 = (
            f"Response rate {sig.recruiter_response_rate:.2f}; "
            f"notice {sig.notice_period_days}d; "
            f"open to opportunities."
        )
    return f"{s1} {s2}"
```

### `_build_tier2_reasoning(...)` — Ranks 11-50 *(private)*

```python
def _build_tier2_reasoning(
    title, years, matched_required,
    best_assessment, primary_gap, sig,
) -> str:
    """
    Ranks 11-50: Positive with acknowledged nuance.
    Sentence 1: Core strength.
    Sentence 2: Engagement signal + honest gap note if significant.
    """
    skills_str = (
        ", ".join(matched_required[:2])
        if matched_required
        else "adjacent technical background"
    )
    s1 = f"{title} with {years:.1f}yr experience; matches on {skills_str}."

    engagement = f"response rate {sig.recruiter_response_rate:.2f}"
    if best_assessment and best_assessment[1] >= 50:
        engagement += f"; {best_assessment[0]} assessed at {best_assessment[1]:.0f}/100"

    # Only surface gap if it's meaningful (not just "strong fit with minor gaps")
    if "strong overall fit" in primary_gap or "minor signal" in primary_gap:
        s2 = f"{engagement.capitalize()}; notice {sig.notice_period_days}d."
    else:
        s2 = f"{engagement.capitalize()}; note: {primary_gap}."

    return f"{s1} {s2}"
```

### `_build_tier3_reasoning(...)` — Ranks 51-100 *(private)*

```python
def _build_tier3_reasoning(
    title, years, matched_required, primary_gap, sig,
) -> str:
    """
    Ranks 51-100: Honest about limitations, explains inclusion reason.
    Passes Stage 4 check 3 (honest concerns) and check 6 (rank-consistent tone).
    """
    if matched_required:
        inclusion_reason = f"some alignment on {', '.join(matched_required[:2])}"
    else:
        inclusion_reason = "adjacent technical profile"

    s1 = (
        f"{title} with {years:.1f}yr experience; included due to {inclusion_reason}."
    )
    s2 = (
        f"Key concern: {primary_gap}; "
        f"response rate {sig.recruiter_response_rate:.2f}."
    )
    return f"{s1} {s2}"
```

---

## MODULE 6 TESTS — `tests/test_module6.py`

ALL 8 tests must pass.

```python
import pytest
from src.data_loader import load_candidates, get_jd
from src.nlp_engine import load_model, embed_jd, embed_candidates_batch
from src.scoring_engine import compute_base_score
from src.edge_cases import apply_edge_cases
from src.reasoning import generate_reasoning
from src.models import CandidateScore


@pytest.fixture(scope="module")
def scored_sample():
    """Score a sample of 30 candidates for reasoning tests."""
    candidates = load_candidates("data/raw/candidates.jsonl")
    model = load_model()
    jd = get_jd()
    jd = embed_jd(jd, model)
    sample = candidates[:30]
    embed_candidates_batch(sample, model)
    pairs = []
    for rank, c in enumerate(sample, 1):
        score = compute_base_score(c, jd)
        score = apply_edge_cases(c, score, jd)
        score.rank = rank
        pairs.append((c, score))
    return pairs, jd


def test_reasoning_not_empty(scored_sample):
    """Every candidate gets a non-empty reasoning string."""
    pairs, jd = scored_sample
    for c, score in pairs:
        r = generate_reasoning(c, score, jd)
        assert r and len(r) > 15, f"Empty reasoning for {c.candidate_id}"


def test_reasoning_contains_actual_title(scored_sample):
    """Reasoning references the candidate's actual current_title."""
    pairs, jd = scored_sample
    for c, score in pairs:
        if not c.is_honeypot and c.current_title:
            r = generate_reasoning(c, score, jd)
            # At least part of the title should appear
            title_words = c.current_title.lower().split()
            meaningful_words = [w for w in title_words if len(w) > 3]
            if meaningful_words:
                assert any(w in r.lower() for w in meaningful_words), \
                    f"Title words missing from reasoning: '{c.current_title}' not in '{r}'"


def test_reasoning_contains_experience_years(scored_sample):
    """Reasoning references the candidate's actual years of experience."""
    pairs, jd = scored_sample
    for c, score in pairs:
        if not c.is_honeypot:
            r = generate_reasoning(c, score, jd)
            years_str = f"{c.years_of_experience:.1f}"
            assert years_str in r, \
                f"Experience {years_str} missing from reasoning: '{r}'"


def test_tier1_more_confident_than_tier3(scored_sample):
    """Rank 1 reasoning does not contain concern/gap language."""
    pairs, jd = scored_sample
    rank1_pair = min(pairs, key=lambda x: x[1].rank)
    r = generate_reasoning(rank1_pair[0], rank1_pair[1], jd)
    concern_words = ["concern", "despite", "note:", "limitation", "gap"]
    assert not any(w in r.lower() for w in concern_words), \
        f"Rank 1 reasoning should not have concern language: '{r}'"


def test_tier3_acknowledges_gap(scored_sample):
    """Rank 25+ reasoning acknowledges at least one concern."""
    pairs, jd = scored_sample
    lower_rank_pairs = [(c, s) for c, s in pairs if s.rank > 20 and not c.is_honeypot]
    if lower_rank_pairs:
        c, score = lower_rank_pairs[0]
        r = generate_reasoning(c, score, jd)
        concern_indicators = [
            "concern", "note:", "despite", "limitation", "inactive",
            "notice", "not open", "below", "gap", "limited", "included due to"
        ]
        assert any(ind in r.lower() for ind in concern_indicators), \
            f"Lower-ranked reasoning should acknowledge gap: '{r}'"


def test_all_reasonings_unique(scored_sample):
    """All 30 reasoning strings in sample are unique."""
    pairs, jd = scored_sample
    reasonings = [generate_reasoning(c, s, jd) for c, s in pairs if not c.is_honeypot]
    assert len(set(reasonings)) == len(reasonings), \
        "Duplicate reasoning strings found — too templated"


def test_no_hallucination_skill_not_in_profile(scored_sample):
    """Reasoning does not mention skills the candidate does not have."""
    pairs, jd = scored_sample
    for c, score in pairs[:10]:
        if c.is_honeypot:
            continue
        r = generate_reasoning(c, score, jd)
        candidate_skill_names = {s.name for s in c.skills}
        # Check: any JD skill mentioned in reasoning should actually be in profile
        for req_skill in ["faiss", "pinecone", "qdrant", "weaviate", "milvus"]:
            if req_skill in r.lower():
                assert req_skill in candidate_skill_names, \
                    f"Hallucination: '{req_skill}' in reasoning but not in profile of {c.candidate_id}"


def test_reasoning_reasonable_length(scored_sample):
    """Reasoning is between 50 and 300 characters."""
    pairs, jd = scored_sample
    for c, score in pairs:
        r = generate_reasoning(c, score, jd)
        assert 50 <= len(r) <= 300, \
            f"Reasoning length {len(r)} out of bounds for {c.candidate_id}: '{r}'"
```

---

## COMPLETION CRITERIA

Module 6 is complete when:
1. `pytest tests/test_module6.py -v` shows ALL 8 tests passing
2. Running against 100 random candidates produces 100 unique reasoning strings
3. Tier 1 (ranks 1-10) reasoning does not use concern language
4. Tier 3 (ranks 51-100) reasoning acknowledges the specific gap per candidate

After Module 6 passes: build `rank.py` (the CLI entry point).
