# src/reasoning.py (temporary stub for Module 5 testing)
# Will be replaced with full implementation in Module 6.
from src.models import CandidateProfile, CandidateScore, JobDescription


def generate_reasoning(candidate: CandidateProfile, score: CandidateScore, jd: JobDescription) -> str:
    return f"Stub: {candidate.current_title}, {candidate.years_of_experience:.1f}yr, rank {score.rank}."
