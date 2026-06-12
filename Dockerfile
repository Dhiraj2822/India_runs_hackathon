# Dockerfile — India Runs Track 1
# Stage 3: Judges run your code in this container.
# BGE model is downloaded at BUILD time (has network) so that
# rank.py runs at EXEC time with NO network and finishes in <5 minutes.

FROM python:3.12-slim

WORKDIR /app

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── spaCy model (downloaded at build time — has network) ─────────────────────
RUN python -m spacy download en_core_web_sm

# ── BGE embedding model (downloaded at build time — ~133MB) ──────────────────
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('BAAI/bge-small-en-v1.5'); \
m.eval(); \
e = m.encode('test', normalize_embeddings=True); \
print(f'BGE model cached, shape: {e.shape}')"

# ── Copy project source (no data files — judges provide candidates.jsonl) ────
COPY src/ ./src/
COPY config.py .
COPY rank.py .
COPY validate_submission.py .
COPY candidate_schema.json .

# ── Verify import health ──────────────────────────────────────────────────────
RUN python -c "\
from src.data_loader import load_candidates, get_jd; \
from src.nlp_engine import load_model; \
from src.ranker import rank_all, write_submission; \
print('All modules import OK')"

# ── Runtime: judges will mount candidates.jsonl and output dir ────────────────
# docker run --rm \
#   -v /path/to/candidates.jsonl:/app/data/raw/candidates.jsonl \
#   -v /path/to/output:/app/output \
#   india-runs-track1 \
#   python rank.py --candidates data/raw/candidates.jsonl --out output/submission.csv

CMD ["python", "rank.py", "--candidates", "data/raw/candidates.jsonl", "--out", "output/submission.csv"]
