"""
CV to JD Mapping System v2 — CV Parser Agent
=============================================
LangGraph node: parse_cv

Responsibilities:
- Extract structured metadata from raw CV text using GPT
- Enrich CV text with structured summary (improves embedding quality)
- Return a ParsedCV Pydantic model via type-safe JSON parsing

FIX vs v1: NO @st.cache_data at this level.
           Caching caused ALL CVs to return the FIRST candidate's metadata.
           The graph handles caching at the workflow level using content-based keys.
"""

import json
import logging
import re

from models.schemas import CVJDState, ParsedCV
from utils.text_extraction import enrich_cv_with_metadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

CV_PARSE_PROMPT = """You are an expert HR analyst. Extract structured information from the CV text below.

Return a VALID JSON object with EXACTLY these keys:
{{
    "candidate_name": "Full name of the candidate (string, 'Unknown' if not found)",
    "candidate_title": "Current or most recent job title (string)",
    "skills": ["skill1", "skill2", ...],
    "experience_years": <total years of professional experience as a number>,
    "education": "Highest qualification, e.g. 'B.E. Mechanical Engineering, IIT Bombay'",
    "work_history": ["Company A – Role (2020–2023)", "Company B – Role (2018–2020)", ...],
    "certifications": ["cert1", "cert2", ...],
    "department": "Inferred department such as HR, Finance, Manufacturing, IT, Legal, etc."
}}

Rules:
- experience_years: sum all years across all roles. Use numeric value only (e.g. 7.5, not "7.5 years").
- skills: list both technical AND soft skills. Maximum 40 items.
- certifications: only professional/industry certifications, not degrees.
- department: infer from the overall profile, not just the title.
- Return ONLY the JSON object. No markdown, no explanation.

CV TEXT:
---
{cv_text}
---

JSON:"""

# ---------------------------------------------------------------------------
# Agent node function
# ---------------------------------------------------------------------------

def parse_cv_node(state: CVJDState) -> CVJDState:
    """
    LangGraph node: parse_cv

    Reads:  state.cv_text, state.cv_filename
    Writes: state.parsed_cv, state.current_step, state.error

    Args:
        state: Current LangGraph CVJDState

    Returns:
        Updated CVJDState with parsed_cv populated
    """
    logger.info("[parse_cv] Parsing CV: %s", state.cv_filename)
    state.current_step = "parse_cv"

    if not state.cv_text or len(state.cv_text.strip()) < 50:
        logger.warning("[parse_cv] CV text too short or empty for: %s", state.cv_filename)
        state.error = f"CV text too short to parse: {state.cv_filename}"
        state.parsed_cv = ParsedCV(
            cv_filename=state.cv_filename,
            raw_text=state.cv_text,
        )
        return state

    # --- GPT call ---
    try:
        parsed_data = _call_llm_for_cv(state.cv_text)
    except Exception as e:
        logger.error("[parse_cv] LLM call failed for %s: %s", state.cv_filename, e)
        state.error = f"CV parse LLM error: {e}"
        state.parsed_cv = ParsedCV(raw_text=state.cv_text)
        return state

    # --- Build ParsedCV (Pydantic validates and coerces) ---
    try:
        parsed_cv = ParsedCV(
            candidate_name=parsed_data.get("candidate_name", "Unknown"),
            candidate_title=parsed_data.get("candidate_title", ""),
            skills=_to_list(parsed_data.get("skills", [])),
            experience_years=parsed_data.get("experience_years", 0),
            education=parsed_data.get("education", ""),
            work_history=_to_list(parsed_data.get("work_history", [])),
            certifications=_to_list(parsed_data.get("certifications", [])),
            department=parsed_data.get("department", ""),
            raw_text=state.cv_text,
        )
    except Exception as e:
        logger.error("[parse_cv] Pydantic model build failed: %s — data: %s", e, parsed_data)
        state.error = f"CV parse model error: {e}"
        state.parsed_cv = ParsedCV(raw_text=state.cv_text)
        return state

    # --- Enrich CV text for better embedding quality ---
    enriched = enrich_cv_with_metadata(
        state.cv_text,
        {
            "skills": parsed_cv.skills,
            "experience_years": parsed_cv.experience_years,
            "education": parsed_cv.education,
            "department": parsed_cv.department,
        }
    )
    parsed_cv.enriched_text = enriched

    state.parsed_cv = parsed_cv
    state.cv_text = enriched  # Use enriched text for downstream embedding

    logger.info(
        "[parse_cv] Done — %s | %s | %.1f yrs | %d skills",
        parsed_cv.candidate_name,
        parsed_cv.candidate_title,
        parsed_cv.experience_years,
        len(parsed_cv.skills),
    )
    return state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_llm_for_cv(cv_text: str) -> dict:
    """
    Call the configured LLM to extract structured data from CV text.
    Returns a dict (may be empty on failure).
    """
    from config.settings import get_llm_client, get_model_name, settings

    client = get_llm_client()
    model = get_model_name()

    # Truncate very long CVs to avoid token limits (keep first 6000 chars)
    truncated_text = cv_text[:6000] if len(cv_text) > 6000 else cv_text

    prompt = CV_PARSE_PROMPT.format(cv_text=truncated_text)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    return _safe_json_parse(raw)


def _safe_json_parse(raw: str) -> dict:
    """
    Parse JSON from LLM response. Tries direct parse first,
    then strips markdown fences, then regex-extracts the JSON block.
    """
    # 1. Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences
    stripped = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 3. Regex: find the first {...} block
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("[parse_cv] Could not parse JSON from LLM response. Raw: %s", raw[:200])
    return {}


def _to_list(value) -> list:
    """Ensure value is a list. Handles strings, None, and non-list types."""
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value:
        return [value]
    return []
