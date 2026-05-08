"""
CV to JD Mapping System v2 — Streamlit UI
==========================================
Three-tab layout:
  Tab 1: Index Job Descriptions (upload JDs → build/update FAISS index)
  Tab 2: Match CVs to JDs      (upload CVs → run pipeline → view reports)
  Tab 3: Analytics Dashboard   (summary view of all results)

Key fixes vs v1:
  - No @st.cache_data on text extraction (caching bug fix)
  - Content-based caching via st.session_state using file hash as key
  - Real-time progress bars during batch processing
  - Scoring breakdown shown with metric cards
  - Azure Blob toggle for remote persistence
"""

import hashlib
import io
import logging
import os
import sys
from datetime import datetime

# Ensure the project root (parent of ui/) is on sys.path so that
# 'utils', 'agents', 'config', 'graph', 'models' are all importable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CV ↔ JD Matcher v2",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports (avoid heavy libs on every rerun)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_embeddings_client():
    from config.settings import get_embeddings_client
    return get_embeddings_client()


@st.cache_resource(show_spinner=False)
def _get_faiss_index():
    """Load or create FAISS index (shared across all users in same session)."""
    from utils.vector_store import FAISSJDIndex
    from config.settings import settings
    try:
        index = FAISSJDIndex.load()
        st.sidebar.success(f"✅ Index loaded ({index.index.ntotal} JDs)")
        return index
    except Exception:
        return FAISSJDIndex()


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_session():
    defaults = {
        "analysis_results": [],     # List[AnalysisReport]
        "indexed_jd_count": 0,
        "cv_cache": {},             # {file_hash: {"text": str, "filename": str}}
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()

# ---------------------------------------------------------------------------
# CSS / Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .verdict-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: bold;
        font-size: 0.9em;
        color: white;
    }
    .main-header {
        font-size: 2.2em;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 4px;
    }
    .sub-header {
        font-size: 1.0em;
        color: #666;
        margin-bottom: 24px;
    }
    .score-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .score-card .value {
        font-size: 1.6em;
        font-weight: 700;
    }
    .score-card .label {
        font-size: 0.75em;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">🎯 CV ↔ JD Mapping System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">v2 · LangGraph Agents · Multi-Provider LLM · Azure Blob Persistence</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider("Top JD matches per CV", min_value=1, max_value=10, value=3)
    use_blob = st.toggle("💾 Sync index to Azure Blob", value=False)

    st.divider()
    st.subheader("📊 Session Stats")
    faiss_index = _get_faiss_index()
    st.metric("JDs in Index", faiss_index.index.ntotal)
    st.metric("CVs Processed", len(st.session_state.analysis_results))

    if st.button("🔄 Reset Index", type="secondary"):
        faiss_index.reset()
        st.session_state.analysis_results = []
        st.session_state.indexed_jd_count = 0
        st.cache_resource.clear()
        st.success("Index cleared.")
        st.rerun()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📁 Step 1: Index JDs", "🔍 Step 2: Match CVs", "📊 Analytics"])


# ============================================================================
# TAB 1 — JD Indexing
# ============================================================================

with tab1:
    st.subheader("Upload & Index Job Descriptions")
    st.info(
        "Upload JD files (PDF or DOCX). Each JD is parsed by GPT, enriched, "
        "and embedded into the FAISS search index. Existing index is preserved — "
        "only new JDs are added."
    )

    jd_files = st.file_uploader(
        "Upload JD files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="jd_uploader",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        index_btn = st.button("⚡ Index JDs", type="primary", disabled=not jd_files)

    if jd_files:
        st.write(f"**{len(jd_files)} file(s) selected:**")
        for f in jd_files:
            st.markdown(f"- `{f.name}` ({f.size / 1024:.1f} KB)")

    if index_btn and jd_files:
        embeddings_client = _get_embeddings_client()
        progress_bar = st.progress(0, text="Preparing...")
        status = st.empty()

        def update_progress(i, total, msg):
            pct = int(i / max(total, 1) * 100)
            progress_bar.progress(pct, text=msg)
            status.markdown(f"Processing: `{msg}`")

        try:
            from agents.jd_analyzer_agent import index_jd_files

            parsed_jds = index_jd_files(
                jd_files=jd_files,
                faiss_index=faiss_index,
                embeddings_client=embeddings_client,
                upload_to_blob=use_blob,
                progress_callback=update_progress,
            )

            progress_bar.progress(100, text="Done!")
            st.session_state.indexed_jd_count += len(parsed_jds)

            st.success(f"✅ Successfully indexed {len(parsed_jds)} JD(s). Total: {faiss_index.index.ntotal}")

            # Show parsed JD summaries
            if parsed_jds:
                st.subheader("Indexed JDs Summary")
                for jd in parsed_jds:
                    with st.expander(f"📋 {jd.job_title or jd.filename}"):
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Min Experience", f"{jd.min_experience_years:.0f} yrs")
                        col_b.metric("Required Skills", len(jd.required_skills))
                        col_c.metric("Department", jd.department or "N/A")
                        if jd.required_skills:
                            st.write("**Required Skills:**", ", ".join(jd.required_skills[:15]))
                        if jd.key_responsibilities:
                            st.write("**Key Responsibilities:**")
                            for r in jd.key_responsibilities[:5]:
                                st.markdown(f"- {r}")

        except Exception as e:
            st.error(f"❌ Indexing failed: {e}")
            logger.exception("JD indexing error")


# ============================================================================
# TAB 2 — CV Matching
# ============================================================================

with tab2:
    st.subheader("Upload CVs & Match to JDs")

    if faiss_index.index.ntotal == 0:
        st.warning("⚠️ No JDs indexed yet. Go to **Step 1** to index JDs first.")
    else:
        st.success(f"✅ Index ready with **{faiss_index.index.ntotal} JDs**. Upload CVs below.")

    cv_files = st.file_uploader(
        "Upload CV files (PDF or DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="cv_uploader",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        match_btn = st.button(
            "🚀 Match CVs",
            type="primary",
            disabled=(not cv_files or faiss_index.index.ntotal == 0),
        )

    if match_btn and cv_files:
        from utils.text_extraction import extract_text_from_uploaded_file
        from graph.workflow import run_batch_pipeline

        embeddings_client = _get_embeddings_client()
        progress_bar = st.progress(0, text="Extracting text...")
        status_box = st.empty()

        # --- Extract text from CVs (content-based cache to fix the v1 caching bug) ---
        cv_items = []
        for cv_file in cv_files:
            file_bytes = cv_file.getvalue()
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            if file_hash in st.session_state.cv_cache:
                # Use cached text (keyed by content hash — never wrong)
                cached = st.session_state.cv_cache[file_hash]
                cv_items.append({"cv_text": cached["text"], "cv_filename": cv_file.name})
            else:
                cv_text = extract_text_from_uploaded_file(cv_file)
                if cv_text:
                    st.session_state.cv_cache[file_hash] = {
                        "text": cv_text, "filename": cv_file.name
                    }
                    cv_items.append({"cv_text": cv_text, "cv_filename": cv_file.name})
                else:
                    st.warning(f"⚠️ Could not extract text from `{cv_file.name}`. Skipping.")

        if cv_items:
            def update_progress(i, total, filename):
                pct = int(i / max(total, 1) * 100)
                progress_bar.progress(pct, text=f"Processing {filename}...")
                status_box.markdown(f"Analyzing: `{filename}`")

            try:
                new_reports = run_batch_pipeline(
                    cv_items=cv_items,
                    faiss_index=faiss_index,
                    embeddings_client=embeddings_client,
                    top_k=top_k,
                    progress_callback=update_progress,
                )

                progress_bar.progress(100, text="Done!")
                st.session_state.analysis_results.extend(new_reports)
                st.success(f"✅ Generated {len(new_reports)} analysis reports.")

            except Exception as e:
                st.error(f"❌ Matching failed: {e}")
                logger.exception("CV matching error")

    # --- Display results ---
    if st.session_state.analysis_results:
        st.divider()
        st.subheader("📋 Analysis Results")

        for report in st.session_state.analysis_results:
            verdict_colors = {
                "Highly Suitable": "#28a745",
                "Potentially Hireable": "#17a2b8",
                "Partially Suitable — Significant Gaps": "#ffc107",
                "Not a Recommended Fit": "#dc3545",
            }
            color = verdict_colors.get(report.final_verdict, "#777")

            with st.expander(
                f"📄 {report.cv_filename} ↔ {report.jd_title or report.jd_filename}"
                f" — {report.confidence_score}%"
            ):
                # Verdict badge + score
                col_score, col_verdict = st.columns([1, 3])
                with col_score:
                    st.metric("Confidence Score", f"{report.confidence_score}%")
                with col_verdict:
                    st.markdown(
                        f'<span class="verdict-badge" style="background-color:{color};">'
                        f"{report.final_verdict}</span>",
                        unsafe_allow_html=True,
                    )

                # Scoring breakdown (4 mini-cards)
                if report.scoring_breakdown:
                    sb = report.scoring_breakdown
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(
                        f'<div class="score-card"><div class="value">'
                        f'{sb.semantic_similarity*100:.0f}%</div>'
                        f'<div class="label">Semantic</div></div>',
                        unsafe_allow_html=True
                    )
                    c2.markdown(
                        f'<div class="score-card"><div class="value">'
                        f'{sb.skill_match_ratio*100:.0f}%</div>'
                        f'<div class="label">Skill Match</div></div>',
                        unsafe_allow_html=True
                    )
                    c3.markdown(
                        f'<div class="score-card"><div class="value">'
                        f'{sb.experience_match_ratio*100:.0f}%</div>'
                        f'<div class="label">Experience</div></div>',
                        unsafe_allow_html=True
                    )
                    c4.markdown(
                        f'<div class="score-card"><div class="value">'
                        f'{sb.education_match*100:.0f}%</div>'
                        f'<div class="label">Education</div></div>',
                        unsafe_allow_html=True
                    )

                st.markdown("")  # Spacer

                # Summary
                st.markdown(f"**Match Summary:** {report.match_summary}")

                # Key insights
                if report.key_hireable_insights:
                    st.markdown("**Key Hireable Insights:**")
                    for insight in report.key_hireable_insights:
                        st.markdown(f"- {insight}")

                # Skills two-column layout
                sk_col1, sk_col2 = st.columns(2)
                with sk_col1:
                    st.success(f"✅ **Matching Skills** ({len(report.matching_skills)})")
                    if report.matching_skills:
                        st.markdown(", ".join(report.matching_skills))
                    else:
                        st.caption("None identified")
                with sk_col2:
                    st.error(f"❌ **Missing Skills** ({len(report.missing_skills)})")
                    if report.missing_skills:
                        st.markdown(", ".join(report.missing_skills))
                    else:
                        st.caption("None — full match!")

        # Export button
        st.divider()
        if st.button("📥 Export to Excel", type="secondary"):
            flat = [r.to_flat_dict() for r in st.session_state.analysis_results]
            df = pd.DataFrame(flat)
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="CV_JD_Analysis")
            buf.seek(0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "⬇️ Download Excel",
                data=buf,
                file_name=f"CV_JD_Analysis_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ============================================================================
# TAB 3 — Analytics Dashboard
# ============================================================================

with tab3:
    st.subheader("📊 Analytics Dashboard")

    results = st.session_state.analysis_results

    if not results:
        st.info("No results yet. Run CV matching in Step 2 first.")
    else:
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        scores = [r.confidence_score for r in results]
        m1.metric("Total Comparisons", len(results))
        m2.metric("Avg Confidence", f"{sum(scores)/len(scores):.1f}%")
        m3.metric("Top Score", f"{max(scores)}%")
        highly_suitable = sum(1 for r in results if r.final_verdict == "Highly Suitable")
        m4.metric("Highly Suitable", highly_suitable)

        st.divider()

        # Verdict distribution
        verdict_counts = {}
        for r in results:
            verdict_counts[r.final_verdict] = verdict_counts.get(r.final_verdict, 0) + 1

        df_verdicts = pd.DataFrame(
            list(verdict_counts.items()), columns=["Verdict", "Count"]
        ).sort_values("Count", ascending=False)
        st.subheader("Verdict Distribution")
        st.bar_chart(df_verdicts.set_index("Verdict"))

        # Results table
        st.subheader("All Results")
        flat = [r.to_flat_dict() for r in results]
        df = pd.DataFrame(flat)
        df_display = df[[
            "CV_Filename", "Candidate_Name", "JD_Title", "Confidence_Score",
            "Final_Verdict", "Semantic_Similarity", "Skill_Match_%"
        ]].sort_values("Confidence_Score", ascending=False)
        st.dataframe(df_display, use_container_width=True)
