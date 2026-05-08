"""
CV to JD Mapping System v2 — LangGraph Workflow
================================================
Wires all agents into a typed StateGraph.

Graph topology:
    START → parse_cv → retrieve_matches → score_matches → generate_reports → END

Each node reads from CVJDState and writes back.
Error short-circuit: if any node sets state.error, subsequent nodes are skipped.

Usage:
    from graph.workflow import build_graph, run_cv_pipeline

    app = build_graph()
    reports = run_cv_pipeline(cv_text="...", cv_filename="John_Doe.pdf")
"""

import logging
from functools import partial
from typing import List, Optional

from langgraph.graph import END, START, StateGraph

from agents.cv_parser_agent import parse_cv_node
from agents.matching_agent import retrieve_matches_node
from agents.scoring_agent import score_matches_node
from agents.report_agent import generate_reports_node
from models.schemas import AnalysisReport, CVJDState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node wrappers with error short-circuit
# ---------------------------------------------------------------------------

def _guarded(node_fn):
    """
    Wrap a node function to skip execution if state.error is already set.
    Allows upstream failures to propagate cleanly through the graph.
    """
    def wrapper(state: CVJDState, **kwargs) -> CVJDState:
        if state.error:
            logger.warning(
                "[workflow] Skipping %s — upstream error: %s",
                node_fn.__name__, state.error
            )
            return state
        return node_fn(state, **kwargs)
    wrapper.__name__ = node_fn.__name__
    return wrapper


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    faiss_index=None,
    embeddings_client=None,
) -> StateGraph:
    """
    Build and compile the LangGraph CV-JD matching state machine.

    Args:
        faiss_index: Pre-loaded FAISSJDIndex to share across invocations.
                     If None, each run loads the index from disk (slower but stateless).
        embeddings_client: Pre-initialized LangChain embeddings client.
                           Shared for efficiency; avoids re-authenticating per CV.

    Returns:
        Compiled LangGraph app (StateGraph.compile())
    """
    graph = StateGraph(CVJDState)

    # --- Register nodes ---
    graph.add_node("parse_cv", _guarded(parse_cv_node))
    graph.add_node(
        "retrieve_matches",
        _guarded(partial(
            retrieve_matches_node,
            faiss_index=faiss_index,
            embeddings_client=embeddings_client,
        )),
    )
    graph.add_node("score_matches", _guarded(score_matches_node))
    graph.add_node("generate_reports", _guarded(generate_reports_node))

    # --- Wire edges ---
    graph.add_edge(START, "parse_cv")
    graph.add_edge("parse_cv", "retrieve_matches")
    graph.add_edge("retrieve_matches", "score_matches")
    graph.add_edge("score_matches", "generate_reports")
    graph.add_edge("generate_reports", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run_cv_pipeline(
    cv_text: str,
    cv_filename: str,
    top_k: Optional[int] = None,
    faiss_index=None,
    embeddings_client=None,
    app=None,
) -> List[AnalysisReport]:
    """
    Run the full CV-JD matching pipeline for a single CV.

    Args:
        cv_text: Raw text extracted from the CV (PDF/DOCX)
        cv_filename: Original filename (used for logging and report metadata)
        top_k: Number of top JD matches to return. Defaults to settings.top_k_matches.
        faiss_index: Pre-loaded FAISSJDIndex (optional; loaded per-run if None)
        embeddings_client: Pre-initialized embeddings client (optional)
        app: Pre-compiled LangGraph app (optional; built fresh if None)

    Returns:
        List of AnalysisReport objects sorted by confidence score (descending)
    """
    from config.settings import settings

    if top_k is None:
        top_k = settings.top_k_matches

    graph_app = app or build_graph(
        faiss_index=faiss_index,
        embeddings_client=embeddings_client,
    )

    initial_state = CVJDState(
        cv_text=cv_text,
        cv_filename=cv_filename,
        top_k=top_k,
    )

    logger.info("[workflow] Running pipeline for: %s (top_k=%d)", cv_filename, top_k)

    try:
        final_state: CVJDState = graph_app.invoke(initial_state)
    except Exception as e:
        logger.error("[workflow] Pipeline crashed for %s: %s", cv_filename, e)
        raise

    if final_state.error:
        logger.warning(
            "[workflow] Pipeline completed with error for %s: %s",
            cv_filename, final_state.error
        )

    reports = final_state.final_reports or []
    logger.info(
        "[workflow] Done — %s → %d reports. Top: %s (%d%%)",
        cv_filename,
        len(reports),
        reports[0].jd_filename if reports else "n/a",
        reports[0].confidence_score if reports else 0,
    )
    return reports


def run_batch_pipeline(
    cv_items: List[dict],
    faiss_index=None,
    embeddings_client=None,
    top_k: Optional[int] = None,
    progress_callback=None,
) -> List[AnalysisReport]:
    """
    Run the pipeline for multiple CVs efficiently.
    Builds the graph once and reuses it for all CVs.

    Args:
        cv_items: List of dicts with keys 'cv_text' and 'cv_filename'
        faiss_index: Pre-loaded FAISSJDIndex
        embeddings_client: Pre-initialized embeddings client
        top_k: Number of top JD matches per CV
        progress_callback: Optional callable(i, total, cv_filename) for UI progress

    Returns:
        Flat list of all AnalysisReport objects across all CVs
    """
    from config.settings import settings
    from utils.vector_store import FAISSJDIndex

    top_k = top_k or settings.top_k_matches

    # Load shared resources once
    if faiss_index is None:
        faiss_index = FAISSJDIndex.load()

    if embeddings_client is None:
        from config.settings import get_embeddings_client
        embeddings_client = get_embeddings_client()

    # Build graph once, reuse across CVs
    app = build_graph(faiss_index=faiss_index, embeddings_client=embeddings_client)

    all_reports: List[AnalysisReport] = []
    total = len(cv_items)

    for i, item in enumerate(cv_items):
        cv_filename = item.get("cv_filename", f"cv_{i}.pdf")
        cv_text = item.get("cv_text", "")

        if progress_callback:
            progress_callback(i, total, cv_filename)

        if not cv_text.strip():
            logger.warning("[workflow] Empty text for %s, skipping.", cv_filename)
            continue

        try:
            reports = run_cv_pipeline(
                cv_text=cv_text,
                cv_filename=cv_filename,
                top_k=top_k,
                app=app,
            )
            all_reports.extend(reports)
        except Exception as e:
            logger.error("[workflow] Batch: pipeline failed for %s: %s", cv_filename, e)
            continue

    logger.info("[workflow] Batch done — %d CVs → %d reports.", total, len(all_reports))
    return all_reports
