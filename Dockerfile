FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model via direct pip link to avoid 404 resolution errors
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz

# Pre-download BGE embedding model into HuggingFace cache (requires network — BUILD time)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu'); \
print('BGE model cached.')"

# Force HuggingFace to run in offline mode at runtime so it doesn't try to check for updates
ENV HF_HUB_OFFLINE=1

# Copy project code
COPY src/ ./src/
COPY config.py rank.py precompute.py ./

# data/raw/candidates.jsonl is NOT included — must be mounted at runtime
# See docker run command below

# Default command: ranking step (no network, uses pre-cached model)
CMD ["python", "rank.py", \
     "--candidates", "./data/raw/candidates.jsonl", \
     "--out", "./submission.csv"]

# ─── How to build and run ────────────────────────────────────────────────────
#
# Build (requires network, done once):
#   docker build -t india-runs-ranker .
#
# Run (no network, uses cached model):
#   docker run \
#     -v /path/to/candidates.jsonl:/app/data/raw/candidates.jsonl \
#     -v /path/to/output:/app/output \
#     --network none \
#     india-runs-ranker \
#     python rank.py --candidates ./data/raw/candidates.jsonl --out ./output/PARTICIPANT_ID.csv
#
# ─────────────────────────────────────────────────────────────────────────────
