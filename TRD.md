# TRD — Technical Requirements Document
## Based on actual dataset, submission_spec.docx, and JD analysis
## Version: FINAL

---

## 1. TECH STACK (DO NOT DEVIATE)

| Component | Choice | Reason |
|-----------|--------|--------|
| Language | Python 3.11 | Stable, modern |
| Embedding model | BAAI/bge-small-en-v1.5 | Best retrieval accuracy at 133MB, CPU-fast |
| NLP pipeline | spaCy en_core_web_sm | Entity extraction, fast |
| Fuzzy matching | rapidfuzz | Skill synonym matching without network |
| Data | pandas 2.2.0 | CSV output, data wrangling |
| Math | numpy 1.26.4 | Embedding math |
| Testing | pytest 8.1.1 | All module tests |
| Date parsing | python-dateutil 2.9.0 | Career date parsing |

---

## 2. requirements.txt (EXACT — COPY AS IS)

```
sentence-transformers==2.7.0
spacy==3.7.4
rapidfuzz==3.9.0
pandas==2.2.0
numpy==1.26.4
python-dateutil==2.9.0
pytest==8.1.1
```

After install run:
```bash
python -m spacy download en_core_web_sm
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5'); print('Model cached.')"
```

The second command pre-downloads the model into HuggingFace cache.
The ranking step (`rank.py`) uses ONLY the local cache — no network.

---

## 3. config.py (COMPLETE — COPY EXACTLY INTO PROJECT ROOT)

```python
"""
config.py
ALL constants live here. No other file may hardcode values.
If a value needs changing, change it here only.
"""

# ─── Paths ────────────────────────────────────────────────────────────────────
CANDIDATES_JSONL = "data/raw/candidates.jsonl"
OUTPUT_CSV        = "submission.csv"
LOGS_PATH         = "logs/errors.log"

# ─── Model ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
SPACY_MODEL     = "en_core_web_sm"
DEVICE          = "cpu"   # GPU forbidden per submission spec

# ─── Pipeline thresholds ──────────────────────────────────────────────────────
TOP_K_STAGE1 = 5000   # Stage 1 fast filter keeps top 5,000
TOP_K_STAGE2 = 200    # Stage 2 semantic scoring keeps top 200
TOP_K_FINAL  = 100    # Final output is exactly 100

# ─── Stage 1 weights (must sum to 1.0) ───────────────────────────────────────
W1_TITLE        = 0.35   # AI/ML title relevance
W1_SKILLS       = 0.30   # Count of required skills in profile
W1_AVAILABILITY = 0.25   # open_to_work + last_active recency
W1_DOMAIN       = 0.10   # Industry/domain in tech/AI

# ─── Stage 2 weights (must sum to 1.0) ───────────────────────────────────────
W2_SEMANTIC      = 0.25  # JD-to-profile semantic similarity
W2_SKILL_QUALITY = 0.30  # Skill proficiency + duration + endorsements + assessment
W2_CAREER_FIT    = 0.20  # Title history + company type + production experience
W2_EXPERIENCE    = 0.10  # Years of experience in JD range
W2_BEHAVIORAL    = 0.10  # Redrob platform behavioral signals
W2_EDUCATION     = 0.05  # Education tier and field relevance

# ─── Proficiency multipliers (for skill quality scoring) ─────────────────────
PROFICIENCY_WEIGHTS = {
    "beginner":     0.25,
    "intermediate": 0.55,
    "advanced":     0.80,
    "expert":       1.00,
}

# ─── Education tier scores ────────────────────────────────────────────────────
EDUCATION_TIER_SCORES = {
    "tier_1": 1.00,
    "tier_2": 0.75,
    "tier_3": 0.50,
    "tier_4": 0.25,
    "unknown": 0.40,
}

# ─── JD: Required and preferred skills (from actual job_description.docx) ─────
JD_REQUIRED_SKILLS = [
    "embeddings", "retrieval", "ranking", "sentence-transformers",
    "vector database", "pinecone", "weaviate", "qdrant", "milvus",
    "faiss", "elasticsearch", "python", "evaluation", "ndcg", "mrr",
    "hybrid search", "dense retrieval", "semantic search",
]

JD_PREFERRED_SKILLS = [
    "llm fine-tuning", "lora", "qlora", "peft", "learning to rank",
    "xgboost", "lightgbm", "open source", "distributed systems",
    "inference optimization", "a/b testing", "bm25", "reranking",
    "opensearch", "recommendation systems",
]

# ─── JD: Experience range ─────────────────────────────────────────────────────
JD_EXPERIENCE_MIN = 5.0   # years
JD_EXPERIENCE_MAX = 9.0   # years

# ─── JD: Preferred notice period ─────────────────────────────────────────────
JD_PREFERRED_NOTICE_DAYS = 30   # <= 30 days preferred

# ─── JD: Preferred locations (lowercase) ─────────────────────────────────────
JD_PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore",
    "bengaluru", "gurgaon", "gurugram",
}

# ─── JD: Disqualified company types (explicitly stated in JD) ─────────────────
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "cognizant technology solutions", "capgemini",
    "hcl", "hcl technologies", "tech mahindra", "mphasis", "hexaware",
    "l&t infotech", "ltimindtree", "mindtree", "persistent systems",
    "niit technologies", "mastech", "syntel", "birlasoft", "cyient",
}

# ─── JD: Honeypot / irrelevant titles (keyword stuffers target these roles) ───
HONEYPOT_TITLES = {
    "marketing manager", "graphic designer", "content writer",
    "accountant", "civil engineer", "mechanical engineer",
    "sales executive", "hr manager", "human resources",
    "customer support", "operations manager", "project manager",
    "supply chain", "procurement manager", "finance manager",
    "legal counsel", "teacher", "lecturer", "nurse", "doctor",
    "interior designer", "fashion designer", "event planner",
}

# ─── Titles that indicate strong AI/ML relevance ─────────────────────────────
AI_ML_TITLES = {
    "machine learning engineer", "ml engineer", "ai engineer",
    "applied scientist", "research engineer", "nlp engineer",
    "data scientist", "recommendation systems engineer",
    "search engineer", "ranking engineer", "information retrieval",
    "applied ml", "deep learning engineer", "retrieval engineer",
    "backend engineer",   # can be relevant with right skills
    "software engineer",  # can be relevant with right skills
    "data engineer",      # can be relevant
    "platform engineer",  # can be relevant
}

# ─── Skill synonym groups (for matching variants of the same skill) ───────────
SKILL_SYNONYMS = {
    "embeddings": [
        "sentence-transformers", "bge", "e5", "openai embeddings",
        "ada", "text embeddings", "dense embeddings", "vector embeddings",
    ],
    "vector database": [
        "pinecone", "weaviate", "qdrant", "milvus", "faiss",
        "opensearch", "elasticsearch", "chromadb", "pgvector",
        "redis vector", "lancedb",
    ],
    "retrieval": [
        "information retrieval", "dense retrieval", "sparse retrieval",
        "hybrid retrieval", "semantic search", "neural search",
        "bm25", "ir", "rag", "retrieval augmented generation",
    ],
    "ranking": [
        "learning to rank", "ltr", "reranking", "re-ranking",
        "candidate ranking", "pointwise ranking", "listwise ranking",
    ],
    "evaluation": [
        "ndcg", "mrr", "map", "precision@k", "recall@k",
        "a/b testing", "ab testing", "offline evaluation",
        "relevance evaluation", "ir evaluation",
    ],
    "python": ["python3", "py", "pytorch", "numpy", "pandas"],
    "llm fine-tuning": [
        "lora", "qlora", "peft", "instruction tuning", "rlhf",
        "fine-tuning", "finetuning", "sft", "dpo",
    ],
    "recommendation systems": [
        "recommender system", "collaborative filtering",
        "matrix factorization", "two-tower model",
    ],
}

# ─── Skill family groups (partial credit when family matches) ─────────────────
SKILL_FAMILY_GROUPS = {
    "vector_db_family": [
        "pinecone", "weaviate", "qdrant", "milvus", "faiss",
        "opensearch", "elasticsearch", "chromadb", "pgvector",
    ],
    "embedding_model_family": [
        "sentence-transformers", "bge", "e5", "ada",
        "openai embeddings", "cohere embeddings", "instructor",
    ],
    "llm_family": [
        "gpt", "llama", "mistral", "gemini", "claude", "palm",
        "falcon", "phi", "qwen",
    ],
    "ml_framework_family": [
        "pytorch", "tensorflow", "jax", "keras",
    ],
    "tree_model_family": [
        "xgboost", "lightgbm", "catboost", "sklearn", "gradient boosting",
    ],
}

# ─── JD text (embedded — no file read needed during ranking) ──────────────────
JD_TEXT = """
Senior AI/ML Engineer — Founding Team. Redrob AI. Pune / Noida, India.
Series A AI-native talent intelligence platform. 5-9 years total experience.

We are building India's most intelligent recruiting AI. You will design and
own the candidate ranking and retrieval layer that powers everything we do.

What you will build:
- End-to-end embedding-based candidate retrieval (sentence-transformers, BGE, E5)
- Production vector database integration (Pinecone, Weaviate, Qdrant, Milvus, FAISS)
- Ranking models: hybrid retrieval, dense + sparse fusion, learning to rank
- Evaluation frameworks: NDCG, MRR, MAP, A/B testing infrastructure
- Strong Python throughout

Preferred experience:
- LLM fine-tuning with LoRA, QLoRA, PEFT
- HR-tech or marketplace product experience
- Distributed systems, inference optimization
- Open-source AI/ML contributions

We explicitly do NOT want:
- Candidates whose entire career is at consulting firms (TCS, Infosys, Wipro,
  Accenture, Cognizant, Capgemini, HCL) without product company experience
- Candidates who switch every 1.5 years chasing titles
- "AI experience" of 12 months using LangChain to call GPT-4 APIs
- Pure research background with no production deployment
- Primary expertise in computer vision, speech, robotics, or audio

Ideal candidate:
- 6-8 years total, 4-5 in applied ML at product companies
- Shipped an end-to-end ranking, search, or recommendation system to real users
- Active on platform, responsive to recruiters
- Notice period under 30 days preferred
- Located in Pune, Noida, Hyderabad, Mumbai, Delhi, or Bangalore — or willing to relocate
"""
```

---

## 4. Staging Budget (time allocation for 100K candidates)

| Stage | Input | Output | Time budget |
|-------|-------|--------|-------------|
| Data loading | 100K JSONL | 100K CandidateProfile | ~15s |
| Stage 1 fast filter | 100K profiles | Top 5K | ~5s |
| Stage 2 embedding | Top 5K | 5K with embeddings | ~90s |
| Stage 2 scoring | 5K scored | Top 200 | ~10s |
| Edge cases | 200 candidates | 200 with adjustments | ~2s |
| Ranking + reasoning | 200 | Top 100 with reasoning | ~5s |
| Write + validate | 100 | submission.csv | ~3s |
| **Total** | | | **~130s (2.2 min)** |

Well within the 5-minute limit.
