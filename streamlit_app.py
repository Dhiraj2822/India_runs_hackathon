"""
streamlit_app.py — Interactive candidate ranking demo.
Designed to run on Streamlit Community Cloud or locally.
Loads sample_candidates.json (50 candidates) and allows running the ranking pipeline live.
Also displays the pre-calculated submission.csv.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from config import (
    EMBEDDING_MODEL,
    W1_AVAILABILITY,
    W1_DOMAIN,
    W1_SKILLS,
    W1_TITLE,
    W2_BEHAVIORAL,
    W2_CAREER_FIT,
    W2_EDUCATION,
    W2_EXPERIENCE,
    W2_SEMANTIC,
    W2_SKILL_QUALITY,
)
from src.data_loader import _parse_candidate, get_jd
from src.nlp_engine import embed_jd, load_model
from src.ranker import rank_all

# ─── Page Settings ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob AI — Candidate Ranker",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom Premium CSS (Glassmorphism & Gradients) ───────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Gradient Banner */
    .header-container {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.3);
    }
    
    /* Glassmorphic Metric Cards */
    .custom-metric {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        color: white;
    }
    
    /* Premium Candidate Glass Cards */
    .candidate-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.6);
        border-left: 6px solid #4f46e5;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .candidate-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08);
        background: #ffffff;
    }
    
    /* Dark Theme Support for Candidate Cards */
    @media (prefers-color-scheme: dark) {
        .candidate-card {
            background: rgba(30, 30, 34, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-left: 6px solid #6366f1;
            color: #f3f4f6;
        }
        .candidate-card:hover {
            background: rgba(30, 30, 34, 0.85);
        }
    }
    
    /* Custom Badge elements */
    .badge {
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.5rem;
    }
    .badge-rank {
        background: rgba(245, 158, 11, 0.15);
        color: #d97706;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-score {
        background: rgba(16, 185, 129, 0.15);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-id {
        background: rgba(79, 70, 229, 0.15);
        color: #4f46e5;
        border: 1px solid rgba(79, 70, 229, 0.3);
    }
    
    @media (prefers-color-scheme: dark) {
        .badge-rank {
            background: rgba(245, 158, 11, 0.25);
            color: #fbbf24;
        }
        .badge-score {
            background: rgba(16, 185, 129, 0.25);
            color: #34d399;
        }
        .badge-id {
            background: rgba(99, 102, 241, 0.25);
            color: #818cf8;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Sidebar configuration ───────────────────────────────────────────────────
st.sidebar.title("Configuration & Info")

st.sidebar.markdown(
    f"""
### Neural Engine
- **Model**: `{EMBEDDING_MODEL}`
- **Device**: `CPU` (No GPU required)
- **Framework**: `sentence-transformers`

### Pipeline Weights
**Stage 1: Fast Filter (W1)**
- Title Match: `{W1_TITLE:.0%}`
- Skills Match: `{W1_SKILLS:.0%}`
- Availability: `{W1_AVAILABILITY:.0%}`
- Industry Domain: `{W1_DOMAIN:.0%}`

**Stage 2: Full Scoring (W2)**
- Semantic Fit: `{W2_SEMANTIC:.0%}`
- Skill Quality: `{W2_SKILL_QUALITY:.0%}`
- Career Fit: `{W2_CAREER_FIT:.0%}`
- Experience Match: `{W2_EXPERIENCE:.0%}`
- Platform Signals: `{W2_BEHAVIORAL:.0%}`
- Education Tier: `{W2_EDUCATION:.0%}`
"""
)

# ─── Main Content Title Banner ────────────────────────────────────────────────
st.markdown(
    """
<div class="header-container">
    <h1 style="margin:0; font-size:2.2rem; font-weight:700; color:white;">Redrob AI — Candidate Ranking System</h1>
    <p style="margin:5px 0 0 0; opacity:0.9; font-size:1.1rem;">India Runs Hackathon Track 1 • Stage Reproduction Sandbox</p>
</div>
""",
    unsafe_allow_html=True,
)

# Tabs
tab_leaderboard, tab_sandbox, tab_reproduce = st.tabs(
    ["Ranked Output", "Live Sandbox", "Methodology & CLI"]
)

# ─── TAB 1: Ranked Output (shows submission.csv) ──────────────────────────────
with tab_leaderboard:
    st.markdown("### Top Ranked Candidates (Full 100K Dataset)")
    st.write(
        "Below is the current `submission.csv` containing the top 100 candidates sorted and filtered by our 2-stage scoring engine."
    )

    csv_path = Path("submission.csv")
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", f"{len(df)}")
            col2.metric("Min Score", f"{df['score'].min():.4f}")
            col3.metric("Max Score", f"{df['score'].max():.4f}")

            st.dataframe(
                df,
                column_config={
                    "candidate_id": st.column_config.TextColumn(
                        "Candidate ID", width="medium"
                    ),
                    "rank": st.column_config.NumberColumn(
                        "Rank", width="small"
                    ),
                    "score": st.column_config.NumberColumn(
                        "Final Score", format="%.4f", width="small"
                    ),
                    "reasoning": st.column_config.TextColumn(
                        "Reasoning & Qualifications", width="large"
                    ),
                },
                width="stretch",
                hide_index=True,
            )
        except Exception as e:
            st.error(f"Error loading submission.csv: {e}")
    else:
        st.info(
            "submission.csv is not available yet. Please run the ranking pipeline on the full dataset or use the Live Sandbox tab."
        )

# ─── TAB 2: Live Sandbox (runs pipeline on sample candidates) ──────────────────
with tab_sandbox:
    st.markdown("### Live Sandbox — Candidate Ranker")
    st.write(
        "Run the complete NLP-powered ranking pipeline live on `sample_candidates.json` (50 candidate profiles)."
    )

    sample_path = Path("data/raw/sample_candidates.json")

    if not sample_path.exists():
        st.error(f"Sample data file not found at: `{sample_path}`")
    else:
        if st.button("Run Live Ranking Pipeline", type="primary"):
            with st.spinner("Initializing models and embedding Job Description..."):
                try:
                    # 1. Load models
                    model = load_model()
                    jd = get_jd()
                    jd = embed_jd(jd, model)

                    # 2. Load and parse candidates
                    with open(sample_path, encoding="utf-8") as f:
                        raw_data = json.load(f)

                    candidates = []
                    for raw in raw_data:
                        parsed = _parse_candidate(raw)
                        if parsed:
                            candidates.append(parsed)

                    st.success(
                        f"Successfully loaded and parsed {len(candidates)} candidates."
                    )

                    with st.spinner("Processing 2-stage ranking & edge cases..."):
                        # 3. Run pipeline
                        results = rank_all(candidates, jd, model)


                    st.markdown("### Ranked Sample Results")

                    for rank_idx, s in enumerate(results, start=1):
                        cand_profile = next(
                            c for c in candidates if c.candidate_id == s.candidate_id
                        )

                        # HTML Card
                        st.markdown(
                            f"""
                            <div class="candidate-card">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                    <div>
                                        <span class="badge badge-rank">Rank #{rank_idx}</span>
                                        <span class="badge badge-score">Score: {s.final_score:.4f}</span>
                                        <span class="badge badge-id">{s.candidate_id}</span>
                                    </div>
                                    <div style="font-weight:600; font-size:1.1rem;">
                                        {cand_profile.name if cand_profile.name else "Anonymized Profile"}
                                    </div>
                                </div>
                                <div style="margin-bottom:8px;">
                                    <strong>Current Role:</strong> {cand_profile.current_title} at {cand_profile.current_company if cand_profile.current_company else "N/A"} ({cand_profile.years_of_experience:.1f} YOE)
                                </div>
                                <div style="font-style: italic; opacity:0.95; padding-left: 10px; border-left: 3px solid #ccc;">
                                    "{s.reasoning}"
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                except Exception as e:
                    st.error(f"Pipeline error during live run: {e}")
                    import traceback

                    st.code(traceback.format_exc())

# ─── TAB 3: Methodology & CLI ─────────────────────────────────────────────────
with tab_reproduce:
    st.markdown("### Pipeline Architecture Overview")
    st.markdown(
        """
The pipeline executes a deterministic two-stage ranking and validation framework designed for extreme throughput, accuracy, and reproducibility.

```mermaid
graph TD
    A[Raw Candidates JSONL] --> B[Stage 1 Fast Filter]
    B -->|Score Structured Fields| C[Top 5000 Candidates]
    C --> D[Stage 2 Semantic Embeddings]
    D -->|BAAI/bge-small-en-v1.5| E[Stage 2 Scoring]
    E -->|Structured + Semantic weights| F[Edge Cases & Penalties]
    F -->|Honeypots/Job Hoppers/Consulting| G[Final Scores Rounded 4dp]
    G -->|Alphabetical Tie-breaker| H[Top 100 Output CSV]
```

#### Key Design Safeguards
1. **Zero Hallucination Reasoning**: The reasoning module uses pure profile facts (actual years of experience, actual titles, actual skills). If a candidate is missing a required skill, it detects it and generates a balanced overview without inventing facts.
2. **Honeypot Disqualification**: Any candidate matching a honeypot template (e.g. Graphic Designer with keyword stuffing) gets flagged and scored 0.0. If more than 10% of the top 100 are honeypots, the validation pipeline triggers an error to prevent submission disqualification.
3. **Deterministic Tie-Breaking**: Ranks are sorted by `(-round(score,4), -raw_score, candidate_id)`. If two candidates share the same rounded 4dp score, the candidate with the higher raw unrounded score wins. If scores are truly identical, the alphabetically lower ID receives the higher rank.
"""
    )

    st.markdown("---")
    st.markdown("### Command Line Interface")
    st.code(
        """
# Step 1: Download and pre-cache models (exempt from 5-minute limit)
python precompute.py

# Step 2: Run ranking pipeline (must finish in under 5 minutes)
python rank.py --candidates data/raw/candidates.jsonl --out submission.csv

# Step 3: Run sanity checker
python validate_submission.py submission.csv
""",
        language="bash",
    )
