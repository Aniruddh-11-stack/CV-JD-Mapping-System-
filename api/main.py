"""
CV to JD Mapping System v2 — FastAPI REST Backend
==================================================
Endpoints:
  POST /jds/index       — Upload + index one or more JD files
  GET  /jds             — List indexed JDs
  POST /cvs/match       — Match a single CV against indexed JDs
  POST /cvs/batch       — Match multiple CVs in one request
  GET  /health          — Liveness probe

Authentication:
  Optional API key via X-API-Key header (set API_KEY_HEADER in .env)

Usage:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from config.settings import settings
from models.schemas import AnalysisReport
from utils.vector_store import FAISSJDIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared resources (loaded once at startup)
# ---------------------------------------------------------------------------

_faiss_index: Optional[FAISSJDIndex] = None
_embeddings_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared resources on startup; clean up on shutdown."""
    global _faiss_index, _embeddings_client

    logger.info("Loading embeddings client...")
    from config.settings import get_embeddings_client
    _embeddings_client = get_embeddings_client()

    logger.info("Loading FAISS index...")
    try:
        _faiss_index = FAISSJDIndex.load()
        logger.info("Index loaded: %d JDs", _faiss_index.index.ntotal)
    except Exception:
        logger.warning("No existing index found. Starting with empty index.")
        _faiss_index = FAISSJDIndex()

    yield

    logger.info("Shutting down CV-JD API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CV ↔ JD Mapping System v2",
    version="2.0.0",
    description="LangGraph-powered CV to Job Description matching API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API Key auth (optional)
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(_api_key_header)):
    """Validate API key if configured; pass through if not configured."""
    if not settings.api_key_header:
        return  # No auth configured
    if api_key != settings.api_key_header:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key.",
        )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class IndexJDsResponse(BaseModel):
    message: str
    indexed_count: int
    total_in_index: int
    job_titles: List[str]


class ListJDsResponse(BaseModel):
    total: int
    jds: List[dict]


class MatchCVRequest(BaseModel):
    cv_text: str
    cv_filename: str = "uploaded_cv.pdf"
    top_k: int = 3


class BatchMatchRequest(BaseModel):
    cv_items: List[dict]   # Each: {"cv_text": str, "cv_filename": str}
    top_k: int = 3


class HealthResponse(BaseModel):
    status: str
    jds_in_index: int
    llm_provider: str
    embedding_model: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check — returns system status."""
    return HealthResponse(
        status="ok",
        jds_in_index=_faiss_index.index.ntotal if _faiss_index else 0,
        llm_provider=settings.llm_provider,
        embedding_model=(
            settings.azure_openai_embedding_deployment
            if settings.llm_provider == "azure_openai"
            else settings.openai_embedding_model
        ),
    )


@app.post("/jds/index", response_model=IndexJDsResponse, tags=["JD Indexing"])
async def index_jds(
    files: List[UploadFile] = File(...),
    upload_to_blob: bool = False,
    _: None = Depends(verify_api_key),
):
    """
    Upload and index one or more JD files (PDF or DOCX).
    Appends to the existing index without replacing it.
    """
    from agents.jd_analyzer_agent import index_jd_files
    from utils.text_extraction import extract_text_from_bytes

    # Validate file types
    for f in files:
        ext = f.filename.rsplit(".", 1)[-1].lower()
        if ext not in ("pdf", "docx"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {f.filename}. Only PDF and DOCX allowed."
            )

    # Build file-like objects with in-memory bytes
    class _BytesFile:
        def __init__(self, content: bytes, name: str):
            self._bytes = content
            self.name = name

        def getvalue(self):
            return self._bytes

    byte_files = []
    for f in files:
        content = await f.read()
        byte_files.append(_BytesFile(content, f.filename))

    try:
        parsed_jds = index_jd_files(
            jd_files=byte_files,
            faiss_index=_faiss_index,
            embeddings_client=_embeddings_client,
            upload_to_blob=upload_to_blob,
        )
    except Exception as e:
        logger.exception("JD indexing failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    return IndexJDsResponse(
        message=f"Successfully indexed {len(parsed_jds)} JD(s).",
        indexed_count=len(parsed_jds),
        total_in_index=_faiss_index.index.ntotal,
        job_titles=[jd.job_title for jd in parsed_jds],
    )


@app.get("/jds", response_model=ListJDsResponse, tags=["JD Indexing"])
def list_jds(_: None = Depends(verify_api_key)):
    """List all JDs currently in the index."""
    jds = []
    for meta in _faiss_index.jd_metadata:
        jds.append({
            "filename": meta.get("filename", ""),
            "job_title": meta.get("job_title", ""),
            "department": meta.get("department", ""),
            "min_experience_years": meta.get("min_experience_years", 0),
            "required_skills_count": len(meta.get("required_skills", [])),
        })
    return ListJDsResponse(total=len(jds), jds=jds)


@app.post("/cvs/match", response_model=List[AnalysisReport], tags=["CV Matching"])
def match_cv(
    request: MatchCVRequest,
    _: None = Depends(verify_api_key),
):
    """
    Match a single CV (raw text) against all indexed JDs.
    Returns the top-K AnalysisReport objects.
    """
    if _faiss_index.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No JDs indexed. POST to /jds/index first.")

    if not request.cv_text.strip():
        raise HTTPException(status_code=400, detail="cv_text cannot be empty.")

    from graph.workflow import run_cv_pipeline

    try:
        reports = run_cv_pipeline(
            cv_text=request.cv_text,
            cv_filename=request.cv_filename,
            top_k=request.top_k,
            faiss_index=_faiss_index,
            embeddings_client=_embeddings_client,
        )
    except Exception as e:
        logger.exception("CV matching failed")
        raise HTTPException(status_code=500, detail=f"Matching failed: {e}")

    return reports


@app.post("/cvs/batch", response_model=List[AnalysisReport], tags=["CV Matching"])
def match_cvs_batch(
    request: BatchMatchRequest,
    _: None = Depends(verify_api_key),
):
    """
    Match multiple CVs against indexed JDs.
    Each item in cv_items must have 'cv_text' and 'cv_filename'.
    """
    if _faiss_index.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No JDs indexed. POST to /jds/index first.")

    if not request.cv_items:
        raise HTTPException(status_code=400, detail="cv_items cannot be empty.")

    from graph.workflow import run_batch_pipeline

    try:
        reports = run_batch_pipeline(
            cv_items=request.cv_items,
            faiss_index=_faiss_index,
            embeddings_client=_embeddings_client,
            top_k=request.top_k,
        )
    except Exception as e:
        logger.exception("Batch matching failed")
        raise HTTPException(status_code=500, detail=f"Batch matching failed: {e}")

    return reports
