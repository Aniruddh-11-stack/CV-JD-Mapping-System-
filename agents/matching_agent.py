"""
CV to JD Mapping System v2 — Matching Agent
============================================
LangGraph node: retrieve_matches

Responsibilities:
- Load the FAISS JD index (from disk or Azure Blob)
- Embed the enriched CV text and run similarity search
- Apply experience-based pre-filter (from prompt_tweaking.py)
- Return top-K MatchResult objects ranked by cosine similarity

Design:
- Uses FAISSJDIndex from utils/vector_store.py
- Pre-filter: only keep JDs where cv_exp >= jd_min_exp (soft filter, not hard cutoff)
- Falls back gracefully if index is empty or not found
"""

import logging
from typing import List, Optional

from models.schemas import CVJDState, MatchResult, ScoringBreakdown

logger = logging.getLogger(__name__)


def retrieve_matches_node(
    state: CVJDState,
    faiss_index=None,     # Injected by graph; None = load from disk
    embeddings_client=None,
) -> CVJDState:
    """
    LangGraph node: retrieve_matches

    Reads:  state.cv_text (enriched), state.parsed_cv, state.top_k
    Writes: state.candidate_matches, state.current_step, state.error

    Args:
        state: Current LangGraph CVJDState
        faiss_index: Pre-loaded FAISSJDIndex (optional; loaded from disk if None)
        embeddings_client: LangChain embeddings (optional; uses settings if None)

    Returns:
        Updated CVJDState with candidate_matches populated
    """
    logger.info("[retrieve_matches] Starting retrieval for: %s", state.cv_filename)
    state.current_step = "retrieve_matches"

    # --- Load index ---
    index = faiss_index or _load_index()
    if index is None or index.index.ntotal == 0:
        logger.warning("[retrieve_matches] FAISS index is empty or unavailable.")
        state.error = "JD index is empty. Please index JDs first."
        return state

    # --- Run similarity search ---
    from config.settings import settings

    query_text = state.cv_text  # Already enriched by cv_parser_agent
    retrieval_k = settings.retrieval_k  # Fetch more than top_k (e.g. 25) to allow filtering

    try:
        raw_results = index.search(
            query_text=query_text,
            top_k=retrieval_k,
            embeddings_client=embeddings_client,
        )
    except Exception as e:
        logger.error("[retrieve_matches] FAISS search failed: %s", e)
        state.error = f"Retrieval failed: {e}"
        return state

    if not raw_results:
        logger.warning("[retrieve_matches] No results returned from FAISS search.")
        state.candidate_matches = []
        return state

    # --- Experience pre-filter (from prompt_tweaking.py) ---
    candidate_exp = state.parsed_cv.experience_years if state.parsed_cv else 0.0
    filtered = _apply_experience_filter(raw_results, candidate_exp)

    # If filter is too aggressive (removes everything), fall back to unfiltered
    if not filtered:
        logger.info(
            "[retrieve_matches] Experience filter removed all candidates "
            "(cv_exp=%.1f). Using unfiltered results.", candidate_exp
        )
        filtered = raw_results

    # --- Build MatchResult objects ---
    matches: List[MatchResult] = []
    top_k = min(state.top_k, len(filtered))

    for score, meta in filtered[:top_k]:
        match = MatchResult(
            jd_filename=meta.get("filename", "unknown.pdf"),
            jd_title=meta.get("job_title", ""),
            similarity_score=round(float(score), 4),
            jd_text=meta.get("text", ""),
            # Scoring details filled in by scoring_agent
            scoring=ScoringBreakdown(semantic_similarity=float(score)),
        )
        matches.append(match)

    state.candidate_matches = matches

    logger.info(
        "[retrieve_matches] Done — %d matches (raw=%d, after filter=%d)",
        len(matches), len(raw_results), len(filtered)
    )
    return state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_index():
    """Attempt to load FAISSJDIndex from local disk (then blob fallback)."""
    from utils.vector_store import FAISSJDIndex
    from config.settings import settings

    try:
        index = FAISSJDIndex.load(local_path=settings.faiss_index_path, try_blob_first=False)
        logger.info("[retrieve_matches] Index loaded: %d JDs", index.index.ntotal)
        return index
    except FileNotFoundError:
        logger.warning("[retrieve_matches] Local index not found. Trying Azure Blob...")
        try:
            return FAISSJDIndex.load(try_blob_first=True)
        except Exception as e:
            logger.error("[retrieve_matches] Blob load also failed: %s", e)
            return None
    except Exception as e:
        logger.error("[retrieve_matches] Index load failed: %s", e)
        return None


def _apply_experience_filter(
    raw_results: list,
    candidate_exp: float,
    exp_key: str = "min_experience_years",
) -> list:
    """
    Keep JDs where candidate meets the minimum experience requirement.
    Soft filter: JDs with no experience requirement are always included.

    Args:
        raw_results: List of (score, metadata) tuples from FAISS search
        candidate_exp: CV candidate's total years of experience
        exp_key: Metadata key for JD minimum experience

    Returns:
        Filtered list of (score, metadata) tuples
    """
    filtered = []
    for score, meta in raw_results:
        jd_min_exp = _parse_exp(meta.get(exp_key, 0))
        if jd_min_exp == 0 or candidate_exp >= jd_min_exp:
            filtered.append((score, meta))
        else:
            logger.debug(
                "[retrieve_matches] Filtered out JD '%s' — requires %.1f yrs, candidate has %.1f",
                meta.get("filename", "?"), jd_min_exp, candidate_exp
            )
    return filtered


def _parse_exp(value) -> float:
    """Safely parse experience value to float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        import re
        match = re.search(r"\d+\.?\d*", value)
        return float(match.group()) if match else 0.0
    return 0.0
