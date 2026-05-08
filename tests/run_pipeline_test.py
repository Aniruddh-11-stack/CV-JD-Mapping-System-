"""
CV to JD Mapping System v2 — End-to-End Pipeline Test
======================================================
Tests the full pipeline:
  1. Extract text from DOCX test files
  2. Index 3 Job Descriptions into FAISS (via Azure OpenAI embeddings)
  3. Match 3 CVs against the index via LangGraph pipeline
  4. Print formatted match results + confidence scores

Run:
    python tests/run_pipeline_test.py
"""

import json
import os
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TEST_DOCS = PROJECT_ROOT / "data" / "test_docs"

# ── Helpers ───────────────────────────────────────────────────────────────────
def sep(char="─", w=72): print(char * w)
def section(title):
    print(); sep("═"); print(f"  {title}"); sep("═")

# ── Load modules ──────────────────────────────────────────────────────────────
print("\n🔧 Loading modules...")
from config.settings import settings
from utils.text_extraction import extract_text_from_path
from utils.vector_store import FAISSJDIndex
from graph.workflow import run_cv_pipeline

# ── Locate test files ─────────────────────────────────────────────────────────
JD_FILES = sorted(TEST_DOCS.glob("JD_*.docx"))
CV_FILES = sorted(TEST_DOCS.glob("CV_*.docx"))

if not JD_FILES:
    sys.exit(f"❌ No JD files found in {TEST_DOCS}")
if not CV_FILES:
    sys.exit(f"❌ No CV files found in {TEST_DOCS}")

print(f"  ✅ Found {len(JD_FILES)} JD files, {len(CV_FILES)} CV files")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Extract JD text and build FAISS index
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 1: EXTRACTING & INDEXING JOB DESCRIPTIONS")

jd_texts, jd_metas = [], []

for jd_path in JD_FILES:
    print(f"\n  📄 {jd_path.name}")
    text = extract_text_from_path(str(jd_path))
    if not text or len(text) < 50:
        print(f"     ⚠️  Skipping — could not extract text.")
        continue
    print(f"     Extracted {len(text):,} characters")
    jd_texts.append(text)
    jd_metas.append({"filename": jd_path.name, "text": text})

print(f"\n  🔢 Embedding {len(jd_texts)} JDs via Azure OpenAI ({settings.azure_openai_embedding_deployment})...")
t0 = time.time()
index = FAISSJDIndex()
index.add_jds(jd_texts=jd_texts, jd_metadata_list=jd_metas)
elapsed = time.time() - t0
total = index.index.ntotal
print(f"  ✅ Indexed {total} JDs in {elapsed:.1f}s")

index_path = str(PROJECT_ROOT / settings.faiss_index_path)
index.save_local(path=index_path)
print(f"  💾 FAISS index saved → {index_path}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Match each CV against the JD index
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 2: MATCHING CVs → LangGraph PIPELINE")

all_results = []   # list of (cv_name, List[AnalysisReport])

for cv_path in CV_FILES:
    sep()
    print(f"\n  👤 CV: {cv_path.name}")
    cv_text = extract_text_from_path(str(cv_path))
    if not cv_text or len(cv_text) < 50:
        print(f"  ⚠️  Skipping — could not extract text.")
        continue
    print(f"  Extracted {len(cv_text):,} characters")
    print(f"  🚀 Running pipeline  (parse → retrieve → score → report)...")

    t0 = time.time()
    try:
        reports = run_cv_pipeline(
            cv_text=cv_text,
            cv_filename=cv_path.name,
            faiss_index=index,
            top_k=settings.top_k_matches,
        )
        elapsed = time.time() - t0
        print(f"  ✅ Pipeline done in {elapsed:.1f}s  →  {len(reports)} report(s)")
    except Exception as exc:
        import traceback
        print(f"  ❌ Pipeline error: {exc}")
        traceback.print_exc()
        continue

    all_results.append((cv_path.name, reports))

    # ── Per-CV results ─────────────────────────────────────────────────────────
    print(f"\n  ┌─ RESULTS FOR: {cv_path.stem}")

    for rank, rpt in enumerate(reports, 1):
        score  = rpt.confidence_score           # 0–100 int
        grade  = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
        bd     = rpt.scoring_breakdown

        print(f"  │")
        print(f"  │  #{rank}  {grade}  {rpt.jd_filename.replace('_', ' ').replace('.docx', '')}")
        print(f"  │      Confidence   : {score}%")
        print(f"  │      Verdict      : {rpt.final_verdict}")
        if bd:
            print(f"  │      Semantic     : {bd.semantic_similarity * 100:.1f}%")
            print(f"  │      Skill match  : {bd.skill_match_ratio * 100:.1f}%")
            print(f"  │      Experience   : {bd.experience_match_ratio * 100:.1f}%")
        if rpt.matching_skills:
            skills_str = ", ".join(rpt.matching_skills[:5])
            print(f"  │      ✅ Top skills : {skills_str}")
        if rpt.missing_skills:
            gaps_str = ", ".join(rpt.missing_skills[:3])
            print(f"  │      ❌ Gaps       : {gaps_str}")
        if rpt.match_summary:
            summary = rpt.match_summary[:120].replace("\n", " ")
            print(f"  │      📝 Summary    : {summary}...")

    print(f"  └{'─' * 67}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Save JSON results
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 3: SAVING RESULTS")

results_file = PROJECT_ROOT / "data" / "test_results.json"
payload = []
for cv_name, reports in all_results:
    payload.append({
        "cv": cv_name,
        "matches": [r.model_dump() for r in reports],
    })

with open(results_file, "w") as fh:
    json.dump(payload, fh, indent=2, default=str)

print(f"\n  ✅ Full results saved → {results_file}")

sep("═")
print(f"\n  🎉 TEST COMPLETE — {len(all_results)}/{len(CV_FILES)} CVs processed")
sep("═")
print()
