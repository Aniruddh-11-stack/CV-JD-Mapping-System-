"""
CV to JD Mapping System v2 — Demo Results
==========================================
Demonstrates the full pipeline output using:
  - Real DOCX text extraction (mammoth)
  - Keyword-based TF-IDF similarity (no API key needed)
  - Realistic scoring breakdown matching actual production output format

This demo is designed to show what the system produces with real Azure
OpenAI credentials — the output structure is identical.

Run:
    python tests/demo_results.py
"""

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
TEST_DOCS = PROJECT_ROOT / "data" / "test_docs"

def sep(c="─", w=72): print(c * w)
def section(t): print(); sep("═"); print(f"  {t}"); sep("═")

# ── Text extraction ───────────────────────────────────────────────────────────
def extract_docx(path: str) -> str:
    import mammoth
    with open(path, "rb") as f:
        result = mammoth.extract_raw_text(f)
    return result.value.strip()

# ── TF-IDF similarity (no API needed) ────────────────────────────────────────
STOP = set("a an the and or but in on at to for of with is are was were be been "
           "being have has had do does did will would could should may might "
           "i we you they he she it this that these those".split())

def tokenise(text: str):
    tokens = re.findall(r"[a-z][a-z0-9+#\-\.]{1,}", text.lower())
    return [t for t in tokens if t not in STOP and len(t) > 2]

def tfidf_vector(tokens, vocab):
    tf = Counter(tokens)
    total = max(len(tokens), 1)
    vec = {}
    for w in vocab:
        vec[w] = tf.get(w, 0) / total
    return vec

def cosine(v1, v2):
    keys = set(v1) & set(v2)
    num = sum(v1[k] * v2[k] for k in keys)
    d1 = math.sqrt(sum(x**2 for x in v1.values()))
    d2 = math.sqrt(sum(x**2 for x in v2.values()))
    if d1 == 0 or d2 == 0:
        return 0.0
    return num / (d1 * d2)

# ── Skill keyword matching ────────────────────────────────────────────────────
SKILL_PATTERNS = [
    "python","sql","pytorch","tensorflow","scikit","xgboost","lightgbm",
    "langchain","langgraph","openai","azure","kubernetes","docker","mlflow",
    "fastapi","spark","databricks","kafka","faiss","power bi","tableau",
    "pandas","numpy","hugging face","transformers","rag","fine-tuning","peft",
    "onnx","tensorrt","kubeflow","mlops","ci/cd","terraform","redis",
    "postgresql","synapse","data factory","nlp","llm","deep learning",
    "machine learning","data engineering","feature store","model serving",
]

def extract_skills(text: str):
    text_lower = text.lower()
    return [s for s in SKILL_PATTERNS if s in text_lower]

# ── Scoring logic ─────────────────────────────────────────────────────────────
def compute_scores(cv_text, jd_text):
    cv_toks = tokenise(cv_text)
    jd_toks = tokenise(jd_text)
    vocab = list(set(cv_toks) | set(jd_toks))
    semantic = round(cosine(tfidf_vector(cv_toks, vocab), tfidf_vector(jd_toks, vocab)), 3)

    cv_skills = set(extract_skills(cv_text))
    jd_skills = set(extract_skills(jd_text))
    matching = cv_skills & jd_skills
    missing  = jd_skills - cv_skills
    skill_ratio = len(matching) / max(len(jd_skills), 1)

    # Experience: look for years of experience
    cv_years = [int(m) for m in re.findall(r"(\d+)\+?\s*year", cv_text.lower())]
    jd_years = [int(m) for m in re.findall(r"(\d+)\+?\s*year", jd_text.lower())]
    exp_ratio = min(max(cv_years, default=3) / max(max(jd_years, default=3), 1), 1.0)

    # Education: degree match
    edu_keywords = ["phd","m.tech","mtech","msc","mba","b.tech","btech","bsc","b.com"]
    cv_edu = [e for e in edu_keywords if e in cv_text.lower()]
    edu_ratio = min(len(cv_edu) / 2.0, 1.0)

    # Weighted overall (matches production weights)
    overall = round(
        0.40 * semantic * 100 +
        0.30 * skill_ratio * 100 +
        0.20 * exp_ratio * 100 +
        0.10 * edu_ratio * 100,
        1
    )

    return {
        "semantic": semantic,
        "skill_ratio": round(skill_ratio, 3),
        "exp_ratio": round(exp_ratio, 3),
        "edu_ratio": round(edu_ratio, 3),
        "overall": int(overall),
        "matching_skills": sorted(matching),
        "missing_skills": sorted(missing),
    }

def verdict(score: int) -> str:
    if score >= 85: return "Strongly Recommended"
    if score >= 70: return "Recommended"
    if score >= 50: return "Conditionally Recommended"
    return "Not Recommended"

def grade(score: int) -> str:
    return "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"

# ══════════════════════════════════════════════════════════════════════════════
print("\n🔧 Loading text extraction (mammoth / python-docx)...")

JD_FILES = sorted(TEST_DOCS.glob("JD_*.docx"))
CV_FILES = sorted(TEST_DOCS.glob("CV_*.docx"))

if not JD_FILES or not CV_FILES:
    sys.exit(f"❌  Run the test doc creator first: data/test_docs/ is empty.")

print(f"   ✅  {len(JD_FILES)} JDs  |  {len(CV_FILES)} CVs  found in data/test_docs/\n")

# ══════════════════════════════════════════════════════════════════════════════
section("STEP 1 — TEXT EXTRACTION FROM DOCX FILES")

jd_corpus, cv_corpus = {}, {}

for f in JD_FILES:
    text = extract_docx(str(f))
    jd_corpus[f.stem] = text
    print(f"  📄 {f.name:<45} {len(text):>6,} chars extracted")

print()
for f in CV_FILES:
    text = extract_docx(str(f))
    cv_corpus[f.stem] = text
    print(f"  👤 {f.name:<45} {len(text):>6,} chars extracted")

# ══════════════════════════════════════════════════════════════════════════════
section("STEP 2 — CV ↔ JD SCORING  (TF-IDF + skill overlap + experience)")

all_results = {}
report_rows = []

for cv_name, cv_text in cv_corpus.items():
    scores_by_jd = {}
    for jd_name, jd_text in jd_corpus.items():
        scores_by_jd[jd_name] = compute_scores(cv_text, jd_text)

    ranked = sorted(scores_by_jd.items(), key=lambda x: x[1]["overall"], reverse=True)
    all_results[cv_name] = ranked

    sep()
    print(f"\n  👤  CV: {cv_name.replace('_', ' ')}")
    print(f"\n  ┌── TOP {len(ranked)} JD MATCHES ────────────────────────────────────────")

    for rank, (jd_name, sc) in enumerate(ranked, 1):
        g = grade(sc["overall"])
        v = verdict(sc["overall"])
        print(f"  │")
        print(f"  │  #{rank}  {g}  {jd_name.replace('_', ' ')}")
        print(f"  │      Overall score  : {sc['overall']}%      {v}")
        print(f"  │      Semantic sim   : {sc['semantic']*100:.1f}%")
        print(f"  │      Skill match    : {sc['skill_ratio']*100:.1f}%  "
              f"({len(sc['matching_skills'])}/{len(sc['matching_skills'])+len(sc['missing_skills'])} skills)")
        print(f"  │      Experience fit : {sc['exp_ratio']*100:.1f}%")
        print(f"  │      Education fit  : {sc['edu_ratio']*100:.1f}%")
        if sc["matching_skills"]:
            top5 = ", ".join(sc["matching_skills"][:5])
            print(f"  │      ✅ Matching     : {top5}")
        if sc["missing_skills"]:
            gaps = ", ".join(sc["missing_skills"][:3])
            print(f"  │      ❌ Gaps         : {gaps}")

        report_rows.append({
            "CV": cv_name.replace("_", " "),
            "JD": jd_name.replace("_", " "),
            "Rank": rank,
            "Overall %": sc["overall"],
            "Verdict": v,
            "Semantic %": round(sc["semantic"] * 100, 1),
            "Skill Match %": round(sc["skill_ratio"] * 100, 1),
            "Experience %": round(sc["exp_ratio"] * 100, 1),
            "Matching Skills": ", ".join(sc["matching_skills"]),
            "Missing Skills": ", ".join(sc["missing_skills"]),
        })

    print(f"  └{'─'*67}")

# ══════════════════════════════════════════════════════════════════════════════
section("STEP 3 — SUMMARY TABLE (Top-1 match per CV)")

print(f"\n  {'CV':<35} {'Best JD Match':<30} {'Score':>6}  {'Verdict'}")
sep()
for cv_name, ranked in all_results.items():
    best_jd, best_sc = ranked[0]
    g = grade(best_sc["overall"])
    print(f"  {cv_name.replace('_',' '):<35} "
          f"{best_jd.replace('_',' '):<30} "
          f"{best_sc['overall']:>5}%  "
          f"{g} {verdict(best_sc['overall'])}")

# ══════════════════════════════════════════════════════════════════════════════
section("STEP 4 — SAVING RESULTS")

out_json = PROJECT_ROOT / "data" / "test_results.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as fh:
    json.dump(report_rows, fh, indent=2)
print(f"\n  ✅  {len(report_rows)} result rows saved → {out_json}")

sep("═")
print(f"\n  🎉  DEMO COMPLETE — {len(cv_corpus)} CVs × {len(jd_corpus)} JDs = {len(report_rows)} results")
print(f"\n  NOTE: Production pipeline uses Azure OpenAI GPT-4.1-mini for deeper")
print(f"        NLP analysis (key_hireable_insights, match_summary, enriched scoring).")
print(f"        This demo shows the same output structure using TF-IDF similarity.")
sep("═")
print()
