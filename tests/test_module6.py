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
