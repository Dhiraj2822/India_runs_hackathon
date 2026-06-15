"""
validate_quality.py
Run this after rank.py to check if your ranking is actually good.
It checks technical correctness + ranking quality + edge cases.

Usage:
    python validate_quality.py --submission YOUR_ID.csv --candidates data/raw/candidates.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import date

# Force UTF-8 for Windows terminals
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import pandas as pd


def load_candidates_map(path):
    print(f"Loading candidates from {path}...")
    cmap = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line.strip())
            cmap[c["candidate_id"]] = c
    print(f"  {len(cmap):,} candidates loaded.\n")
    return cmap


def check_technical(df, cmap):
    print("=" * 55)
    print("SECTION 1 — TECHNICAL CORRECTNESS")
    print("=" * 55)
    issues = []

    # Row count
    status = "[PASS]" if len(df) == 100 else "[FAIL]"
    print(f"{status} Row count: {len(df)} (must be 100)")
    if len(df) != 100:
        issues.append("Wrong row count")

    # Columns
    expected = ["candidate_id", "rank", "score", "reasoning"]
    status = "[PASS]" if list(df.columns) == expected else "[FAIL]"
    print(f"{status} Columns: {list(df.columns)}")
    if list(df.columns) != expected:
        issues.append("Wrong columns")

    # Ranks 1-100 unique
    ranks = sorted(df["rank"].tolist())
    status = "[PASS]" if ranks == list(range(1, 101)) else "[FAIL]"
    print(f"{status} Ranks 1-100 unique: {ranks == list(range(1, 101))}")
    if ranks != list(range(1, 101)):
        issues.append("Ranks not 1-100 or duplicates exist")

    # Candidate IDs unique
    status = "[PASS]" if df["candidate_id"].nunique() == 100 else "[FAIL]"
    print(f"{status} Candidate IDs unique: {df['candidate_id'].nunique() == 100}")

    # All candidate IDs exist in dataset
    bad_ids = [cid for cid in df["candidate_id"] if cid not in cmap]
    status = "[PASS]" if not bad_ids else "[FAIL]"
    print(f"{status} All IDs exist in dataset: {not bad_ids}")
    if bad_ids:
        print(f"    BAD IDs: {bad_ids[:5]}")
        issues.append(f"Invalid candidate IDs: {bad_ids[:3]}")

    # Scores non-increasing
    sorted_df = df.sort_values("rank")
    scores = sorted_df["score"].tolist()
    non_increasing = all(scores[i] >= scores[i + 1] - 1e-9 for i in range(len(scores) - 1))
    status = "[PASS]" if non_increasing else "[FAIL]"
    print(f"{status} Scores non-increasing: {non_increasing}")
    if not non_increasing:
        for i in range(len(scores) - 1):
            if scores[i] < scores[i + 1] - 1e-9:
                print(f"    Violation at rank {i+1} ({scores[i]}) → rank {i+2} ({scores[i+1]})")
                break
        issues.append("Scores are not non-increasing")

    # Score differentiation
    unique_scores = df["score"].nunique()
    status = "[PASS]" if unique_scores >= 70 else "[WARN]"
    print(f"{status} Unique score values: {unique_scores}/100 (want ≥ 70)")
    if unique_scores < 50:
        issues.append("Scores too similar — model not differentiating")

    # Score range
    score_min = df["score"].min()
    score_max = df["score"].max()
    score_range = score_max - score_min
    status = "[PASS]" if score_range > 0.15 else "[WARN]"
    print(f"{status} Score range: {score_min:.4f} → {score_max:.4f} (range={score_range:.4f}, want >0.15)")

    # No empty reasoning
    empty_r = df["reasoning"].isna().sum() + (df["reasoning"] == "").sum()
    status = "[PASS]" if empty_r == 0 else "[FAIL]"
    print(f"{status} Empty reasoning strings: {empty_r} (must be 0)")
    if empty_r > 0:
        issues.append(f"{empty_r} empty reasoning strings")

    # Reasoning uniqueness
    unique_r = df["reasoning"].nunique()
    status = "[PASS]" if unique_r >= 90 else "[WARN]"
    print(f"{status} Unique reasoning strings: {unique_r}/100 (want ≥ 90)")
    if unique_r < 90:
        issues.append("Reasoning too templated — too many identical strings")

    print(f"\nTechnical result: {'[PASS]' if not issues else '[FAIL]'}")
    if issues:
        for i in issues:
            print(f"  [FAIL] {i}")
    return issues


def check_top10_quality(df, cmap):
    print("\n" + "=" * 55)
    print("SECTION 2 — TOP 10 QUALITY (50% of your score)")
    print("=" * 55)
    print("These must all be genuine ML/AI engineers. Check each one:\n")

    sorted_df = df.sort_values("rank")
    issues = []
    ai_titles = {
        "ml engineer", "machine learning", "ai engineer", "applied scientist",
        "research engineer", "nlp engineer", "data scientist",
        "recommendation", "search engineer", "ranking engineer",
        "retrieval engineer", "backend engineer", "software engineer",
        "data engineer", "platform engineer"
    }
    consulting = {
        "tcs", "infosys", "wipro", "accenture", "cognizant",
        "capgemini", "hcl", "tech mahindra"
    }

    for _, row in sorted_df.head(10).iterrows():
        cid = row["candidate_id"]
        c = cmap.get(cid, {})
        p = c.get("profile", {})
        sig = c.get("redrob_signals", {})
        skills = [s["name"] for s in c.get("skills", [])[:6]]

        title = p.get("current_title", "UNKNOWN")
        years = p.get("years_of_experience", 0)
        company = p.get("current_company", "UNKNOWN")
        open_work = sig.get("open_to_work_flag", False)
        last_active = sig.get("last_active_date", "2000-01-01")
        response = sig.get("recruiter_response_rate", 0)
        notice = sig.get("notice_period_days", 999)

        # Check days inactive
        try:
            days_inactive = (date.today() - date.fromisoformat(last_active)).days
        except Exception:
            days_inactive = 999

        # Title relevance
        title_ok = any(t in title.lower() for t in ai_titles)
        # Consulting check
        is_consulting_only = company.lower() in consulting

        # Flags
        flags = []
        if not title_ok:
            flags.append(f"[WARN] NON-AI TITLE: {title}")
        if is_consulting_only:
            flags.append(f"[WARN] CONSULTING COMPANY: {company}")
        if not open_work:
            flags.append("[WARN] NOT OPEN TO WORK")
        if days_inactive > 60:
            flags.append(f"[WARN] INACTIVE {days_inactive} DAYS")
        if notice > 90:
            flags.append(f"[WARN] NOTICE {notice} DAYS")

        marker = "[PASS]" if not flags else "[WARN]"
        print(f"Rank {row['rank']:3d} {marker}  score={row['score']:.4f}")
        print(f"         {title} | {years:.1f}yr | {company}")
        print(f"         Open={open_work} | Inactive={days_inactive}d | Response={response:.2f} | Notice={notice}d")
        print(f"         Skills: {skills}")
        print(f"         Reasoning: {row['reasoning'][:90]}...")
        for flag in flags:
            print(f"         {flag}")
            if "NON-AI TITLE" in flag or "CONSULTING" in flag:
                issues.append(f"Rank {row['rank']}: {flag}")
        print()

    print(f"Top-10 result: {'[CLEAN]' if not issues else '[PROBLEMS FOUND]'}")
    if issues:
        for i in issues:
            print(f"  [FAIL] {i}")
        print("\n  If top 10 has non-AI titles or consulting-only companies,")
        print("  your Stage 1 title scoring or Stage 2 career_fit scoring is wrong.")
    return issues


def check_honeypots(df, cmap):
    print("\n" + "=" * 55)
    print("SECTION 3 — HONEYPOT CHECK")
    print("=" * 55)
    print("Honeypot rate > 10% in top 100 = auto-disqualified.\n")

    honeypots_in_top100 = []
    for _, row in df.iterrows():
        cid = row["candidate_id"]
        c = cmap.get(cid, {})
        skills = c.get("skills", [])

        # Check: expert + 0 months (3+ skills)
        expert_zero = [
            s for s in skills
            if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
        ]
        if len(expert_zero) >= 3:
            honeypots_in_top100.append((row["rank"], cid, f"expert+0months: {len(expert_zero)} skills"))
            continue

        # Check: claimed experience vs career history
        years = c.get("profile", {}).get("years_of_experience", 0)
        career_months = sum(ch.get("duration_months", 0) for ch in c.get("career_history", []))
        if (years * 12) - career_months > 60:
            honeypots_in_top100.append((row["rank"], cid,
                f"exp inflation: claimed {years}yr but history shows {career_months/12:.1f}yr"))

    if honeypots_in_top100:
        print(f"[FAIL] HONEYPOTS FOUND IN TOP 100: {len(honeypots_in_top100)}")
        for rank, cid, reason in honeypots_in_top100:
            print(f"   Rank {rank}: {cid} — {reason}")
        rate = len(honeypots_in_top100) / 100
        print(f"\n   Rate: {rate:.0%} (limit is 10%)")
        if rate > 0.10:
            print("   [FAIL] EXCEEDS 10% LIMIT — WOULD BE DISQUALIFIED")
        else:
            print("   [WARN] Below limit but still bad — fix edge cases")
    else:
        print("[PASS] No obvious honeypots detected in top 100")
    return honeypots_in_top100


def check_bottom10(df, cmap):
    print("\n" + "=" * 55)
    print("SECTION 4 — BOTTOM 10 SPOT CHECK")
    print("=" * 55)
    print("Ranks 91-100 should have honest, gap-acknowledging reasoning.\n")

    sorted_df = df.sort_values("rank", ascending=False)
    for _, row in sorted_df.head(10).iterrows():
        c = cmap.get(row["candidate_id"], {})
        p = c.get("profile", {})
        title = p.get("current_title", "UNKNOWN")
        print(f"Rank {row['rank']:3d}  score={row['score']:.4f}  {title}")
        print(f"         Reasoning: {row['reasoning'][:100]}...")
        print()


def check_score_spread(df):
    print("\n" + "=" * 55)
    print("SECTION 5 — SCORE DISTRIBUTION")
    print("=" * 55)

    sorted_df = df.sort_values("rank")
    scores = sorted_df["score"].tolist()

    buckets = {
        ">0.80 (top tier)": sum(1 for s in scores if s > 0.80),
        "0.60-0.80 (strong)": sum(1 for s in scores if 0.60 <= s <= 0.80),
        "0.40-0.60 (moderate)": sum(1 for s in scores if 0.40 <= s < 0.60),
        "<0.40 (weak)": sum(1 for s in scores if s < 0.40),
    }

    for label, count in buckets.items():
        bar = "=" * count
        print(f"  {label:25s}: {count:3d}  {bar}")

    print(f"\n  Rank 1  score: {scores[0]:.4f}")
    print(f"  Rank 10 score: {scores[9]:.4f}")
    print(f"  Rank 50 score: {scores[49]:.4f}")
    print(f"  Rank 100 score: {scores[99]:.4f}")

    # Ideal: rank 1 > 0.75, rank 100 < 0.65, good spread
    if scores[0] < 0.70:
        print("\n  [WARN] Rank 1 score is low (<0.70). Top candidate may not be truly excellent.")
    if scores[99] > 0.75:
        print("\n  [WARN] Rank 100 score is high (>0.75). Scoring not differentiating enough.")
    if scores[0] - scores[99] < 0.10:
        print("\n  [FAIL] Score range too small (<0.10). All candidates scoring similarly.")
    else:
        print(f"\n  [PASS] Score range: {scores[0] - scores[99]:.4f}")


def print_final_verdict(tech_issues, top10_issues, honeypots):
    print("\n" + "=" * 55)
    print("FINAL VERDICT")
    print("=" * 55)

    all_clear = not tech_issues and not top10_issues and not honeypots
    if all_clear:
        print("[PASS] LOOKS GOOD — ready to submit")
        print("\nBefore submitting:")
        print("  1. Confirm filename = your Hack2Skill participant ID")
        print("  2. Run: python validate_submission.py YOUR_ID.csv")
        print("  3. Review top 10 one more time manually")
        print("  4. Submit early — earlier timestamp wins tiebreaks")
    else:
        print("[FAIL] ISSUES FOUND — fix before submitting\n")
        if tech_issues:
            print("Technical fixes needed:")
            for i in tech_issues:
                print(f"  [FAIL] {i}")
        if top10_issues:
            print("\nTop-10 quality fixes needed:")
            for i in top10_issues:
                print(f"  [WARN] {i}")
        if honeypots:
            print("\nHoneypot fixes needed:")
            print(f"  [FAIL] {len(honeypots)} honeypots in top 100 — fix edge_cases.py")

        print("\nWhere to look:")
        if any("NON-AI TITLE" in str(i) for i in top10_issues):
            print("  → src/scoring_engine.py _s1_title_relevance() — title filter too loose")
        if any("CONSULTING" in str(i) for i in top10_issues):
            print("  → src/edge_cases.py _ec_consulting_only() — consulting filter not firing")
        if honeypots:
            print("  → src/edge_cases.py _detect_honeypot() — honeypot detection not working")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, help="Path to your submission CSV")
    parser.add_argument("--candidates", default="data/raw/candidates.jsonl")
    args = parser.parse_args()

    df = pd.read_csv(args.submission)
    cmap = load_candidates_map(args.candidates)

    tech_issues = check_technical(df, cmap)
    top10_issues = check_top10_quality(df, cmap)
    honeypots = check_honeypots(df, cmap)
    check_bottom10(df, cmap)
    check_score_spread(df)
    print_final_verdict(tech_issues, top10_issues, honeypots)


if __name__ == "__main__":
    main()
