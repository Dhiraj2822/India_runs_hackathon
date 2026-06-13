# Redrob AI Candidate Ranking System

This repository implements the complete candidate ranking pipeline for the India Runs Hackathon (Track 1) under the 5-minute CPU runtime constraints.

The system processes 100,000 candidate profiles through a deterministic, high-throughput, and hallucination-free ranking pipeline using local NLP models.

---

## Quick Start & Reproduction

### Prerequisites

Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### Section 1: Pre-computation (Exempt from 5-minute limit)
Before running the pipeline, execute the precompute script. This script downloads and caches the required spaCy model and the `BAAI/bge-small-en-v1.5` sentence transformer model, and verifies your local environment.
```bash
python precompute.py
```

### Section 2: Ranking Pipeline CLI (Must finish in < 5 minutes)
Once the models are cached, run the main ranking script. This execution operates entirely locally with **no network calls**, **no GPU requirements**, and completes the 100,000 candidates scoring in under 5 minutes on standard CPU:
```bash
python rank.py --candidates data/raw/candidates.jsonl --out submission.csv
```
*Note: For the final submission, please rename `submission.csv` or supply `--out your_registered_participant_id.csv` to match your Hack2Skill registered ID.*

### Section 3: Verify Submission
Validate the generated CSV format using the hackathon's submission validator:
```bash
python validate_submission.py submission.csv
```

---

## Interactive Streamlit Sandbox

An interactive dashboard is available to inspect the pipeline's output, review candidates, and run the ranker dynamically. The Live Sandbox tab allows users to upload custom JSON/JSONL candidate datasets and run the full pipeline in real-time, exporting the results to CSV.

### Run Streamlit App Locally:
```bash
streamlit run streamlit_app.py
```

---

## Docker Reproduction

To replicate the Stage 3 judge execution environment, build the Docker container:

```bash
# Build the container (caches the models at build-time)
docker build -t redrob-ranker .

# Run the ranking pipeline in the isolated container (no network access)
docker run --network none -v $(pwd)/data:/app/data redrob-ranker python rank.py --candidates data/raw/candidates.jsonl --out /app/output.csv
```

---

## Pipeline Architecture

The system uses a highly optimized, two-stage ranking process:

1. **Stage 1: Fast Structured Filter**
   - Scores all 100,000 candidates in ~1.5 seconds based on structured attributes (AI/ML title relevance, exact skill keyword matching, notice period/recency availability, and domain alignment).
   - Retains the top 2,000 candidates for advanced semantic scoring.

2. **Stage 2: Full Scoring & Semantic Alignment**
   - Generates semantic embeddings for the top 2,000 candidates using `BAAI/bge-small-en-v1.5` on the CPU.
   - Combines semantic similarity with structured signals and **Behavioral Multipliers**:
     - `recruiter_response_rate` (availability multiplier)
     - `last_active_date` (recency decay penalty)
     - `notice_period_days` (availability penalty)
     - `open_to_work_flag` (heavy hard-filter multiplier)
     - `github_activity_score` (technical validation boost)

3. **Edge Cases & Guardrails**
   - Applies 15 advanced detectors including MAANG consulting penalties and "AI hype" experience mismatch penalties.
   - **Honeypot Trap Disqualification**: Detects and zeros out scores for temporal inconsistencies (e.g., candidate start year predates company founding) and fabricated expert skills (e.g., 10+ skills with 0 years used).
   - Deterministic tie-breaking: Candidates with identical scores are sorted alphabetically by `candidate_id` ascending.

4. **Hallucination-Free Reasoning**
   - Generates highly dynamic, non-templated explanation sentences that read like human recruiter notes.
   - Sentence structure rotates deterministically by rank to avoid repetitive phrasing.
   - Every sentence is built strictly from factual data retrieved directly from the candidate's profile (actual experience years, actual target skills, actual titles) preventing any AI-generated hallucinations.

---

## Repository Structure

```
.
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Module 1: JSONL loader & schema validator
│   ├── models.py            # Pydantic data models
│   ├── nlp_engine.py        # Module 2: Model loader & BGE embedding encoder
│   ├── scoring_engine.py    # Module 3: Scoring logic for Stage 1 and Stage 2
│   ├── edge_cases.py        # Module 4: 15 penalty & bonus detectors
│   ├── ranker.py            # Module 5: Orchestration, sort logic, CSV output
│   └── reasoning.py         # Module 6: Hallucination-free reasoning generator
│
├── tests/                   # 59 pytest unit and integration tests
├── config.py                # Job description and pipeline constants
├── Dockerfile               # Stage 3 container reproduction setup
├── precompute.py            # Offline model caching script
├── rank.py                  # Main CLI execution script
├── streamlit_app.py         # Streamlit Sandbox interface
├── validate_submission.py   # Hackathon CSV format validator
└── submission_metadata.yaml # Stage 2 code portal metadata file
```
