"""
CV to JD Mapping System v2 — JD Analyzer Agent
===============================================
Used during the JD Indexing phase (not part of the CV matching graph).

Responsibilities:
- Extract structured metadata from JD text using GPT
- Enrich JD text with metadata summary before embedding (improves FAISS retrieval)
- Store parsed JD metadata alongside FAISS vectors for use in scoring

Called from:
- ui/app.py (Tab 1: Index JDs)
- api/main.py (POST /jds/index endpoint)
- graph/workflow.py (optional JD-preprocessing sub-graph)
"""

import json
import logging
import re
from typing import List, Optional

from models.schemas import ParsedJD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

JD_PARSE_PROMPT = """You are an expert job analyst. Extract structured information from the Job Description below.

Return a VALID JSON object with EXACTLY these keys:
{{
    "job_title": "Exact job title from the JD (string)",
    "required_skills": ["skill1", "skill2", ...],
    "nice_to_have_skills": ["skill1", "skill2", ...],
    "min_experience_years": <minimum years of experience required, as a number>,
    "required_education": "e.g. 'B.Tech / B.E. in Mechanical Engineering'",
    "department": "e.g. 'Manufacturing', 'HR', 'Finance', 'IT', 'Legal'",
    "location": "Job location or 'Not specified'",
    "key_responsibilities": [
        "Responsibility 1",
        "Responsibility 2",
        "Up to 8 items"
    ]
}}

Rules:
- required_skills: only skills explicitly stated as mandatory/required. Maximum 25 items.
- nice_to_have_skills: skills listed as preferred/good-to-have. Maximum 10 items.
- min_experience_years: use the lower bound of any range (e.g. "5-8 years" → 5). Use 0 if not specified.
- Return ONLY the JSON. No markdown, no explanation.

JD TEXT:
---
{jd_text}
---

JSON:"""

# ---------------------------------------------------------------------------
# Main function (called from UI / API, not a LangGraph node)
# ---------------------------------------------------------------------------

def analyze_jd(
    jd_text: str,
    filename: str = "",
    embeddings_client=None,
) -> ParsedJD:
    """
    Parse a single JD text into a ParsedJD model.
    Called during JD indexing (not part of CV matching pipeline).

    Args:
        jd_text: Raw JD text extracted from PDF/DOCX
        filename: Source filename (for metadata)
        embeddings_client: Unused here; included for future multi-modal use

    Returns:
        ParsedJD with all fields populated
    """
    if not jd_text or len(jd_text.strip()) < 30:
        logger.warning("[jd_analyzer] JD text too short for: %s", filename)
        return ParsedJD(raw_text=jd_text, filename=filename)

    try:
        parsed_data = _call_llm_for_jd(jd_text)
    except Exception as e:
        logger.error("[jd_analyzer] LLM call failed for %s: %s", filename, e)
        return ParsedJD(raw_text=jd_text, filename=filename)

    try:
        jd = ParsedJD(
            job_title=parsed_data.get("job_title", ""),
            required_skills=_to_list(parsed_data.get("required_skills", [])),
            nice_to_have_skills=_to_list(parsed_data.get("nice_to_have_skills", [])),
            min_experience_years=parsed_data.get("min_experience_years", 0),
            required_education=parsed_data.get("required_education", ""),
            department=parsed_data.get("department", ""),
            location=parsed_data.get("location", ""),
            key_responsibilities=_to_list(parsed_data.get("key_responsibilities", [])),
            raw_text=jd_text,
            filename=filename,
        )
    except Exception as e:
        logger.error("[jd_analyzer] Model build failed for %s: %s", filename, e)
        return ParsedJD(raw_text=jd_text, filename=filename)

    logger.info(
        "[jd_analyzer] Done — '%s' | %d required skills | %.0f+ yrs | %s",
        jd.job_title, len(jd.required_skills), jd.min_experience_years, jd.department
    )
    return jd


def build_jd_metadata_for_index(parsed_jd: ParsedJD) -> dict:
    """
    Build the metadata dict stored alongside each FAISS vector.
    This powers the scoring agent and experience pre-filter.

    Args:
        parsed_jd: ParsedJD model

    Returns:
        Flat dict with all fields serializable to string/int/float/bool
    """
    return {
        "filename": parsed_jd.filename,
        "job_title": parsed_jd.job_title,
        "required_skills": parsed_jd.required_skills,      # List[str]
        "nice_to_have_skills": parsed_jd.nice_to_have_skills,
        "min_experience_years": parsed_jd.min_experience_years,
        "required_education": parsed_jd.required_education,
        "department": parsed_jd.department,
        "location": parsed_jd.location,
        "key_responsibilities": parsed_jd.key_responsibilities,
        "text": parsed_jd.raw_text,                        # Full JD text (for report agent)
    }


def enrich_jd_text_for_embedding(parsed_jd: ParsedJD) -> str:
    """
    Create enriched JD text for embedding. Appending structured metadata
    improves FAISS retrieval quality (same principle as CV enrichment in v1).

    Args:
        parsed_jd: ParsedJD model

    Returns:
        Enriched text string = original JD text + structured summary
    """
    parts = []
    if parsed_jd.job_title:
        parts.append(f"Job Title: {parsed_jd.job_title}")
    if parsed_jd.required_skills:
        parts.append("Required Skills: " + ", ".join(parsed_jd.required_skills))
    if parsed_jd.nice_to_have_skills:
        parts.append("Preferred Skills: " + ", ".join(parsed_jd.nice_to_have_skills))
    if parsed_jd.min_experience_years:
        parts.append(f"Minimum Experience: {parsed_jd.min_experience_years} years")
    if parsed_jd.required_education:
        parts.append(f"Required Education: {parsed_jd.required_education}")
    if parsed_jd.department:
        parts.append(f"Department: {parsed_jd.department}")
    if parsed_jd.key_responsibilities:
        parts.append("Key Responsibilities: " + ". ".join(parsed_jd.key_responsibilities[:4]))

    if parts:
        return parsed_jd.raw_text + "\n\n[Structured JD Summary]\n" + "\n".join(parts)
    return parsed_jd.raw_text


def index_jd_files(
    jd_files: list,
    faiss_index,
    embeddings_client=None,
    upload_to_blob: bool = False,
    progress_callback=None,
) -> List[ParsedJD]:
    """
    Full JD indexing pipeline:
    1. Extract text from each file
    2. Analyze with GPT (ParsedJD)
    3. Enrich text for embedding
    4. Embed and add to FAISS index
    5. Optionally save to Azure Blob

    Args:
        jd_files: List of Streamlit UploadedFile objects or file paths
        faiss_index: FAISSJDIndex instance to add JDs into
        embeddings_client: LangChain embeddings
        upload_to_blob: If True, upload index to Azure Blob after indexing
        progress_callback: Optional callable(int, int, str) for UI progress

    Returns:
        List of ParsedJD objects for all successfully indexed JDs
    """
    from utils.text_extraction import extract_text_from_uploaded_file, extract_text_from_path

    parsed_jds: List[ParsedJD] = []
    texts_to_embed: List[str] = []
    metadata_list: List[dict] = []

    for i, jd_file in enumerate(jd_files):
        # Support both UploadedFile (Streamlit) and file path strings
        if hasattr(jd_file, "getvalue"):
            filename = jd_file.name
            jd_text = extract_text_from_uploaded_file(jd_file)
        else:
            filename = jd_file
            jd_text = extract_text_from_path(jd_file)

        if not jd_text:
            logger.warning("[jd_analyzer] Could not extract text from: %s", filename)
            continue

        if progress_callback:
            progress_callback(i, len(jd_files), f"Analyzing {filename}...")

        # Parse JD
        parsed_jd = analyze_jd(jd_text, filename=filename)
        parsed_jds.append(parsed_jd)

        # Enrich text for better embedding
        enriched_text = enrich_jd_text_for_embedding(parsed_jd)

        # Build FAISS metadata
        metadata = build_jd_metadata_for_index(parsed_jd)

        texts_to_embed.append(enriched_text)
        metadata_list.append(metadata)

    if texts_to_embed:
        if progress_callback:
            progress_callback(len(jd_files), len(jd_files), "Embedding and indexing...")

        faiss_index.add_jds(
            jd_texts=texts_to_embed,
            jd_metadata_list=metadata_list,
            embeddings_client=embeddings_client,
        )
        faiss_index.save(upload_to_blob=upload_to_blob)

        logger.info(
            "[jd_analyzer] Indexed %d JDs. Total in index: %d",
            len(texts_to_embed), faiss_index.index.ntotal
        )

    return parsed_jds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_llm_for_jd(jd_text: str) -> dict:
    from config.settings import get_llm_client, get_model_name

    client = get_llm_client()
    model = get_model_name()

    truncated = jd_text[:6000] if len(jd_text) > 6000 else jd_text
    prompt = JD_PARSE_PROMPT.format(jd_text=truncated)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    return _safe_json_parse(raw)


def _safe_json_parse(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    stripped = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("[jd_analyzer] JSON parse failed. Raw: %s", raw[:200])
    return {}


def _to_list(value) -> list:
    if isinstance(value, list):
        return [str(item).strip() for item in value if item]
    if isinstance(value, str) and value:
        return [value]
    return []
