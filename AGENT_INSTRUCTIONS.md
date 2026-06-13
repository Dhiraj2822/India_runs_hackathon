# AGENT MASTER INSTRUCTIONS
## India Runs Hackathon — Track 1: Intelligent Candidate Discovery & Ranking
## Version: FINAL (based on full dataset + submission_spec.docx analysis)
## ⚠️ READ THIS ENTIRE FILE BEFORE TOUCHING ANY OTHER FILE OR WRITING ANY CODE.

---

## WHAT THIS SYSTEM IS

A Python pipeline that reads 100,000 candidate profiles (JSONL) and a job description,
and produces a CSV file ranking the top 100 candidates.

```
INPUT:  candidates.jsonl  (100,000 JSON objects, one per line)
OUTPUT: submission.csv    (exactly 100 rows, columns: candidate_id,rank,score,reasoning)
```

The system runs on CPU only, uses no network during ranking, finishes in under 5 minutes,
and uses ≤ 16GB RAM.

---

## PRIME DIRECTIVE

**NDCG@10 carries 50% of the total score.** Getting the top 10 candidates right
is the single most important thing in this entire system. Everything else serves that goal.

Formula: `0.50 × NDCG@10  +  0.30 × NDCG@50  +  0.15 × MAP  +  0.05 × P@10`

---

## THE 5-STAGE EVALUATION PIPELINE (know what judges see)

```
Stage 1: Format validation     → auto-rejected if any spec violation
Stage 2: Scoring               → NDCG/MAP computed against hidden ground truth
Stage 3: Code reproduction     → top-N repos run in Docker, 5min/16GB/no-GPU/no-network
Stage 3: Honeypot check        → submissions with honeypot rate >10% in top 100 DISQUALIFIED
Stage 4: Manual review         → reasoning quality, git history authenticity, methodology
Stage 5: Defend-your-work      → 30-min video call with Redrob engineers
```

You must build this system well enough to survive all 5 stages.

---

## MODULE BUILD ORDER (SACRED — NEVER CHANGE)

```
MODULE 1 → MODULE 2 → MODULE 3 → MODULE 4 → MODULE 5 → MODULE 6 → rank.py
```

Do NOT start any module until the previous module's tests ALL pass.
Do NOT skip a module.
Do NOT merge two modules into one file.

---

## IMMUTABLE CODE RULES

### RULE 1: TESTED FUNCTIONS ARE PERMANENTLY LOCKED
Once a function has a test that passes, that function is IMMUTABLE.

You may NEVER:
- Change its name
- Change its parameters (names, types, order, defaults)
- Change its return type
- Change its internal logic
- Delete it
- Move it to another file

If a new requirement needs different behavior from an existing function:
→ Write a new function with a new name beside it. Never modify the original.

Example:
```python
# LOCKED after tests pass — never touch again
def score_skills(candidate, jd):
    ...

# New requirement comes in → ADD, never edit
def score_skills_weighted(candidate, jd, weights):
    ...
```

### RULE 2: ADDITIVE CHANGES ONLY
When adding a new edge case, feature, or signal:
1. ADD a new private function with a clear name
2. Call it from the orchestrator function AFTER all existing calls
3. NEVER insert a new call in the middle of an existing function body
4. NEVER modify existing private functions

The orchestrator `apply_edge_cases()` in `src/edge_cases.py` has a fixed call order.
New edge cases are appended AT THE END, before the final_score calculation line.
That final_score calculation line is the last statement in the function — always.

### RULE 3: ZERO CODE DUPLICATION BETWEEN MODULES
Each function lives in exactly ONE file.
Other modules import it. They never copy it.

```python
# CORRECT — Module 3 uses nlp_engine function
from src.nlp_engine import cosine_similarity

# WRONG — copying the function into scoring_engine.py
def cosine_similarity(a, b):  # DO NOT DO THIS
    ...
```

Before writing any function, check whether it already exists in another module.
If it exists → import it. Do not recreate it.

### RULE 4: FUNCTION SIGNATURE CONTRACTS
These signatures are fixed. Downstream modules depend on them exactly.
Changing a signature breaks all callers. If a change is needed, create a new function.

```python
# src/data_loader.py
load_candidates(path: str) -> List[CandidateProfile]
get_jd() -> JobDescription

# src/nlp_engine.py
load_model() -> SentenceTransformer
compute_embedding(text: str, model: SentenceTransformer) -> np.ndarray
embed_candidates_batch(candidates: List[CandidateProfile], model: SentenceTransformer) -> List[CandidateProfile]
cosine_similarity(a: np.ndarray, b: np.ndarray) -> float

# src/scoring_engine.py
compute_stage1_score(candidate: CandidateProfile, jd: JobDescription) -> float
compute_base_score(candidate: CandidateProfile, jd: JobDescription) -> CandidateScore

# src/edge_cases.py
apply_edge_cases(candidate: CandidateProfile, score: CandidateScore, jd: JobDescription) -> CandidateScore

# src/ranker.py
rank_all(candidates: List[CandidateProfile], jd: JobDescription, model: SentenceTransformer) -> List[CandidateScore]

# src/reasoning.py
generate_reasoning(candidate: CandidateProfile, score: CandidateScore, jd: JobDescription) -> str
```

### RULE 5: ERROR HANDLING PROTOCOL
If you encounter an error while building:

1. Log to `logs/errors.log` with format:
   `[TIMESTAMP] ERROR | module=X | function=Y | message=Z | traceback=...`
2. Write `# ERROR: [description]` comment at the problem line
3. STOP. Report the error to the user with exact details.
4. DO NOT work around it by changing the architecture.
5. DO NOT silently skip the failing component.
6. DO NOT proceed to the next module with a broken current module.

### RULE 6: TEST BEFORE PROCEEDING
Every module has tests in `tests/test_moduleX.py`.
Run them with: `pytest tests/test_moduleX.py -v`
ALL tests must pass. Not "most". ALL.
A single failing test = do not proceed to next module.

### RULE 7: NO SHORTCUTS FOR SPEED
The submission deadline is July 2. There is time to do this properly.
Do not skip edge cases because they seem minor.
Do not use placeholder/stub implementations and "fix later".
Do not implement "fast approximate" versions of algorithms to save time.
Each module must be production-quality before the next starts.

### RULE 8: NEVER EDIT validate_submission.py
This file is provided by the hackathon organizers. It is the ground truth for format.
Do not modify it. Only run it.

---

## PROJECT STRUCTURE (FIXED)

```
india_runs_track1/
│
├── AGENT_INSTRUCTIONS.md        ← this file
├── TRD.md                       ← tech spec + config.py content
├── BACKEND_SCHEMA.md            ← data models
├── IMPLEMENTATION_PLAN.md       ← build schedule
│
├── modules/
│   ├── MODULE_1_DATA.md         ← data_loader.py spec
│   ├── MODULE_2_NLP.md          ← nlp_engine.py spec
│   ├── MODULE_3_SCORING.md      ← scoring_engine.py spec
│   ├── MODULE_4_EDGE_CASES.md   ← edge_cases.py spec
│   ├── MODULE_5_RANKER.md       ← ranker.py spec
│   └── MODULE_6_REASONING.md   ← reasoning.py spec
│
├── src/
│   ├── __init__.py
│   ├── models.py                ← all dataclasses (from BACKEND_SCHEMA.md)
│   ├── data_loader.py           ← Module 1 output
│   ├── nlp_engine.py            ← Module 2 output
│   ├── scoring_engine.py        ← Module 3 output
│   ├── edge_cases.py            ← Module 4 output
│   ├── ranker.py                ← Module 5 output
│   └── reasoning.py             ← Module 6 output
│
├── tests/
│   ├── test_module1.py
│   ├── test_module2.py
│   ├── test_module3.py
│   ├── test_module4.py
│   ├── test_module5.py
│   └── test_module6.py
│
├── data/
│   ├── raw/
│   │   └── candidates.jsonl     ← 100,000 candidates (real dataset)
│   └── embeddings/              ← cached embeddings (auto-generated)
│
├── output/
│   └── submission.csv           ← FINAL SUBMISSION FILE
│
├── logs/
│   └── errors.log
│
├── rank.py                      ← CLI entry point (built last)
├── config.py                    ← all constants (from TRD.md)
├── requirements.txt
├── submission_metadata.yaml     ← fill in before submitting
├── validate_submission.py       ← DO NOT EDIT — hackathon-provided validator
└── candidate_schema.json        ← DO NOT EDIT — reference only
```

---

## CRITICAL DATASET FACTS (memorise these)

| Fact | Value |
|------|-------|
| File format | JSONL — one JSON object per line |
| Total candidates | 100,000 |
| Output rows | Exactly 100 |
| Output format | candidate_id, rank, score, reasoning |
| Score range | float 0.0 to 1.0 |
| Ranks | integers 1 to 100, each exactly once |
| Score order | Non-increasing (rank 1 has highest score) |
| Tie-breaker | candidate_id ascending (alphabetically lower = lower rank number) |
| Honeypots | ~80 candidates with impossible profiles |
| Entry command | `python rank.py --candidates ./data/raw/candidates.jsonl --out ./submission.csv` |

---

## TRAP TYPES IN THE DATASET (build defences for all 4)

1. **Honeypots** (~80): Impossible profile facts. e.g., 8yr exp at 3yr-old company.
   Defence: timeline contradiction check + skill duration contradiction check.

2. **Keyword Stuffers**: AI skills in skills section, but career is "Marketing Manager".
   Defence: title-to-skills coherence check. Career history must support skill claims.

3. **Plain-Language Tier 5s**: Glowing summary text, weak structured profile.
   Defence: weight structured data (skills, career_history, redrob_signals) over summary text.

4. **Behavioral Twins**: High engagement signals, wrong domain entirely.
   Defence: behavioral signals are a MULTIPLIER on skill/career fit, not a replacement.

---

## AGENT WORK SEQUENCE

```
Step 0: Read ALL module docs before starting any code.
Step 1: pip install -r requirements.txt
Step 2: python -m spacy download en_core_web_sm
Step 3: python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
        # This pre-downloads the model. Must succeed before building Module 2.
Step 4: Build src/models.py (dataclasses from BACKEND_SCHEMA.md)
Step 5: Build Module 1. Run tests. ALL pass. Only then proceed.
Step 6: Build Module 2. Run tests. ALL pass. Only then proceed.
Step 7: Build Module 3. Run tests. ALL pass. Only then proceed.
Step 8: Build Module 4. Run tests. ALL pass. Only then proceed.
Step 9: Build Module 5. Run tests. ALL pass. Only then proceed.
Step 10: Build Module 6. Run tests. ALL pass. Only then proceed.
Step 11: Build rank.py. Run end-to-end.
Step 12: python validate_submission.py submission.csv
         Must print: "Submission is valid."
Step 13: Check runtime < 300 seconds.
Step 14: Fill in submission_metadata.yaml.
```
