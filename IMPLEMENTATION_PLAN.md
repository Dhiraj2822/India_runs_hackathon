# IMPLEMENTATION PLAN
## India Runs Track 1 — Complete Build Schedule
## Deadline: July 2, 2026 | ~25 working days from today

---

## rank.py — ENTRY POINT (Build after all modules pass)

```python
#!/usr/bin/env python3
"""
rank.py — India Runs Hackathon, Track 1: Intelligent Candidate Discovery
Redrob AI × Hack2Skill

Reproduce command:
    python rank.py --candidates ./data/raw/candidates.jsonl --out ./submission.csv

Constraints:
    - No network access during execution
    - No GPU (CPU only)
    - < 5 minutes wall-clock on 16GB RAM machine
    - Output: exactly 100 rows, columns: candidate_id,rank,score,reasoning
"""
import argparse
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redrob AI Candidate Ranking System"
    )
    parser.add_argument(
        "--candidates",
        default="data/raw/candidates.jsonl",
        help="Path to candidates JSONL file (default: data/raw/candidates.jsonl)",
    )
    parser.add_argument(
        "--out",
        default="submission.csv",
        help="Output CSV path (default: submission.csv)",
    )
    args = parser.parse_args()

    # Validate input
    if not Path(args.candidates).exists():
        print(f"ERROR: Candidates file not found: {args.candidates}", file=sys.stderr)
        sys.exit(1)

    t_start = time.time()
    print("=" * 55)
    print("  Redrob AI — Candidate Ranking System")
    print("  India Runs Hackathon, Track 1")
    print("=" * 55)

    # Step 1: Load candidates
    print("\n[1/5] Loading candidates...")
    from src.data_loader import load_candidates, get_jd
    candidates = load_candidates(args.candidates)
    t1 = time.time()
    print(f"      {len(candidates):,} candidates loaded  [{t1 - t_start:.1f}s]")

    # Step 2: Load model
    print("\n[2/5] Loading embedding model (local cache, no network)...")
    from src.nlp_engine import load_model, embed_jd
    model = load_model()
    t2 = time.time()
    print(f"      Model ready  [{t2 - t1:.1f}s]")

    # Step 3: Embed JD
    print("\n[3/5] Embedding job description...")
    jd = get_jd()
    jd = embed_jd(jd, model)
    t3 = time.time()
    print(f"      JD embedded  [{t3 - t2:.1f}s]")

    # Step 4: Run ranking pipeline
    print("\n[4/5] Running 2-stage ranking pipeline...")
    from src.ranker import rank_all, write_submission
    top100 = rank_all(candidates, jd, model)
    t4 = time.time()
    print(f"      Pipeline complete — top {len(top100)} candidates ranked  [{t4 - t3:.1f}s]")

    # Step 5: Write and validate output
    print("\n[5/5] Writing submission and validating...")
    candidates_map = {c.candidate_id: c for c in candidates}
    write_submission(top100, candidates_map, args.out)
    t5 = time.time()

    # Summary
    elapsed = t5 - t_start
    print("\n" + "=" * 55)
    print(f"  Output: {args.out}")
    print(f"  Runtime: {elapsed:.1f}s")
    if elapsed > 300:
        print("  ⚠️  WARNING: Exceeded 5-minute budget!")
        print(f"  ⚠️  Over by: {elapsed - 300:.1f}s")
    else:
        print(f"  Budget remaining: {300 - elapsed:.0f}s")
    print("=" * 55)


if __name__ == "__main__":
    main()
```

---

## BUILD SCHEDULE (25 working days, July 2 deadline)

### Days 1-2: Environment Setup
```bash
# Install everything
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Pre-download embedding model (only network touch before ranking)
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-small-en-v1.5')
print('Model cached:', m)
"

# Create directory structure
mkdir -p src tests data/raw data/embeddings output logs

# Create empty __init__.py
touch src/__init__.py

# Verify candidates file is in place
python -c "
import json
line = open('data/raw/candidates.jsonl').readline()
d = json.loads(line)
print('First candidate:', d['candidate_id'])
print('Schema keys:', list(d.keys()))
"
```
**Done when:** All installs succeed and first candidate prints as `CAND_0000001`.

---

### Days 3-4: src/models.py
Build all dataclasses from BACKEND_SCHEMA.md.
No tests for this file — it is tested indirectly through Module 1.
Completion check: `python -c "from src.models import CandidateProfile; print('OK')"`

---

### Days 5-7: Module 1 — Data Loader
Build `src/data_loader.py` following MODULE_1_DATA.md exactly.
```bash
pytest tests/test_module1.py -v
```
**All 12 tests must pass.**

---

### Days 8-9: Module 2 — NLP Engine
Build `src/nlp_engine.py` following MODULE_2_NLP.md exactly.
```bash
pytest tests/test_module2.py -v
```
**All 9 tests must pass.**

---

### Days 10-12: Module 3 — Scoring Engine
Build `src/scoring_engine.py` following MODULE_3_SCORING.md exactly.
This is the most complex module. Do not rush it.
```bash
pytest tests/test_module3.py -v
```
**All 10 tests must pass.**

---

### Days 13-15: Module 4 — Edge Cases
Build `src/edge_cases.py` following MODULE_4_EDGE_CASES.md exactly.
Honeypot detection is critical — test it carefully.
```bash
pytest tests/test_module4.py -v
```
**All 10 tests must pass.**

---

### Days 16-17: Module 5 — Ranker
Build `src/ranker.py` following MODULE_5_RANKER.md exactly.
```bash
pytest tests/test_module5.py -v
```
**All 9 tests must pass.**
Note: Module 6 (reasoning) must exist as a stub for Module 5 tests to import.
Create a minimal stub first:
```python
# src/reasoning.py (temporary stub for Module 5 testing)
from src.models import CandidateProfile, CandidateScore, JobDescription
def generate_reasoning(candidate, score, jd) -> str:
    return f"Stub reasoning for {candidate.candidate_id} rank {score.rank}."
```
Replace the stub with full implementation in Module 6.

---

### Days 18-19: Module 6 — Reasoning Generator
Build `src/reasoning.py` following MODULE_6_REASONING.md exactly.
Replace the stub with full implementation.
```bash
pytest tests/test_module6.py -v
```
**All 8 tests must pass.**

---

### Day 20: rank.py + End-to-End Test
Write `rank.py` (copy from IMPLEMENTATION_PLAN.md).
Run the full pipeline:
```bash
time python rank.py --candidates ./data/raw/candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv
```
Both must succeed. Time must be under 300 seconds.
Check output manually:
```bash
head -5 submission.csv
wc -l submission.csv   # should be 101 (header + 100 rows)
```

---

### Day 21: Quality Review
Manual inspection of top 10:
- Open submission.csv, look at ranks 1-10
- Are they AI/ML engineers? (Not Marketing Managers)
- Are they open to work?
- Do their reasoning strings reference actual skills?
- Are any of them from the known-honeypot pattern?

Run a quick analysis:
```python
import json, pandas as pd
df = pd.read_csv("submission.csv")
# Load candidates for top 10
candidates_map = {}
with open("data/raw/candidates.jsonl") as f:
    for line in f:
        c = json.loads(line)
        candidates_map[c['candidate_id']] = c

print("=== TOP 10 REVIEW ===")
for _, row in df.head(10).iterrows():
    cid = row['candidate_id']
    c = candidates_map[cid]
    print(f"Rank {row['rank']}: {c['profile']['current_title']} | "
          f"exp={c['profile']['years_of_experience']} | "
          f"open={c['redrob_signals']['open_to_work_flag']} | "
          f"score={row['score']}")
    print(f"  Skills: {[s['name'] for s in c['skills'][:5]]}")
    print(f"  Reasoning: {row['reasoning']}")
    print()
```
If top 10 look wrong (non-ML titles, honeypots, closed candidates) — revisit Module 3 or 4 weights.

---

### Day 22: Sandbox Deployment (REQUIRED for submission)

Deploy to **Streamlit Community Cloud** or **HuggingFace Spaces**.
The `sandbox_link` in `submission_metadata.yaml` is REQUIRED.

**Option A: Streamlit Cloud (recommended)**
```python
# streamlit_app.py
import streamlit as st
import pandas as pd

st.title("Redrob AI — Candidate Ranking System")
st.subheader("India Runs Hackathon Track 1")

st.markdown("""
## Architecture Overview
- **Stage 1**: Fast structured scoring (100K candidates in seconds)
- **Stage 2**: Semantic embedding via BAAI/bge-small-en-v1.5
- **Edge Cases**: 15 detectors including honeypot detection
- **Reasoning**: Per-candidate, non-templated explanations

## Sample Output
""")

# Show submission.csv as a sample
try:
    df = pd.read_csv("submission.csv")
    st.dataframe(df.head(20))
except:
    st.info("submission.csv not available in this demo.")

st.markdown("""
## Reproduce Command
```bash
python rank.py --candidates ./data/raw/candidates.jsonl --out ./submission.csv
```
""")
```
Push to GitHub, connect Streamlit Cloud, get the live URL.

---

### Day 23: Fill submission_metadata.yaml

```yaml
# Fill in all fields — every field is required
team_name: "Your Team Name"
team_members:
  - name: "Your Name"
    email: "your@email.com"
    role: "Solo / Team Lead"

submission_info:
  github_repo: "https://github.com/YOUR_USERNAME/india-runs-track1"
  sandbox_link: "https://your-app.streamlit.app"    # REQUIRED
  reproduce_command: "python rank.py --candidates ./data/raw/candidates.jsonl --out ./submission.csv"
  submission_file: "submission.csv"

technical_info:
  has_network_during_ranking: false     # MUST be false
  uses_gpu_for_inference: false         # MUST be false
  estimated_runtime_seconds: 130
  peak_memory_gb: 4.0
  embedding_model: "BAAI/bge-small-en-v1.5"
  key_libraries: ["sentence-transformers", "spacy", "rapidfuzz", "pandas"]

methodology_summary: |
  Two-stage pipeline. Stage 1 applies fast structured scoring (title relevance,
  required skill count, availability signals) to reduce 100K candidates to 5K.
  Stage 2 runs semantic embedding via BAAI/bge-small-en-v1.5, computes 6-component
  scores (semantic match, skill quality with proficiency/duration/assessment,
  career fit, experience alignment, behavioral signals, education tier), and applies
  15 edge-case detectors including honeypot detection and keyword-stuffer identification.
  Reasoning is generated per-candidate from actual profile facts, with rank-consistent
  tone (confident for top 10, honest about gaps for ranks 51-100).

declarations:
  read_submission_spec: true
  code_is_original_work: true
  ai_tools_used: "Cursor IDE for code completion"
```

---

### Days 24-25: README.md + Final Submission

Write a clear README.md covering:
1. Problem statement (1 paragraph)
2. Architecture diagram (ASCII is fine)
3. Key design decisions (why 2-stage, why this model, why these edge cases)
4. How to install and run
5. What the output looks like (paste first 5 rows of submission.csv)
6. Edge cases handled (list all 15)

Then submit:
1. `python validate_submission.py submission.csv` → must print valid
2. Push all code to GitHub (public repo)
3. Submit on Hack2Skill portal

---

## FULL SUBMISSION CHECKLIST

```
□ GitHub repo is PUBLIC
□ All 6 modules built and tests passing (58 total tests)
□ rank.py runs end-to-end in under 300 seconds
□ python validate_submission.py submission.csv → "Submission is valid."
□ submission.csv has exactly 100 rows, 4 columns
□ No honeypot in top 100 (verified by test_module5)
□ All reasoning strings unique (≥90 of 100)
□ submission_metadata.yaml filled completely
□ sandbox_link is live and accessible
□ has_network_during_ranking: false
□ uses_gpu_for_inference: false
□ README.md explains architecture clearly
□ Git history shows real iteration (not one giant commit)
□ You can explain every design decision (Stage 5 interview)
```

---

## DEFEND-YOUR-WORK PREP (Stage 5 — 30 min video call)

Questions you must be ready to answer:

1. "Why did you choose BAAI/bge-small-en-v1.5 over all-MiniLM-L6-v2?"
   Answer: BGE was specifically optimised for retrieval tasks (not just semantic similarity).
   It outperforms MiniLM on BEIR retrieval benchmarks. For a ranking task like this,
   retrieval-optimised embeddings give better signal.

2. "Why keep 2,000 in Stage 1 instead of 5,000?"
   Answer: The 5-minute CPU wall-clock constraint. BGE embedding on CPU processes ~125 candidates/sec.
   5,000 candidates took ~364 seconds alone, exceeding the 300-second budget. Reducing to 2,000
   brings embedding time to ~146 seconds (total runtime ~170 seconds). Top-100 quality is
   unaffected — Stage 1 already eliminates all non-AI/ML candidates; ranks 2001-5000 are
   borderline profiles that would never make the final 100 anyway.

3. "How do you detect honeypots?"
   Answer: Expert proficiency with 0 months used is the primary signal (impossible in practice).
   Secondary: claimed years_of_experience significantly exceeding total career_history months.
   Tertiary: all JD skills present as expert, no assessments, no endorsements, no duration.

4. "Why is skill quality weighted at 0.30 when semantic is only 0.25?"
   Answer: The dataset has explicit keyword-stuffing traps. A candidate with a summary
   full of AI keywords (high semantic) but no actual skill evidence (proficiency, duration,
   assessment) should rank lower than one with proven skills. The quality signals
   (proficiency × duration × endorsements × assessment) are more trustworthy than
   raw text similarity for this use case.

5. "What would you change with more time?"
   Answer: Learning-to-rank on top of the base scores using a small labelled set.
   Better company-type classification (product vs consulting is binary here but could
   be more nuanced). Richer reasoning via calling the Redrob AI LLM locally.
```
