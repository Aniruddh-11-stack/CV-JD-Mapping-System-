"""
CV to JD Mapping System v2 — Report Agent
==========================================
LangGraph node: generate_reports

Responsibilities:
- Takes scored MatchResult objects and generates detailed GPT analysis reports
- Uses the v1 rubric-based prompt (from prompt_tweaking.py) — enhanced with structured output
- Returns AnalysisReport Pydantic objects with all fields populated
- Handles partial failures gracefully (one JD failure doesn't block others)

GPT Scoring Rubric (from v1 prompt_tweaking.py, refined):
  85–100 → Highly Suitable
  70–84  → Potentially Hireable
  50–69  → Partially Suitable — Significant Gaps
  0–49   → Not a Recommended Fit
"""

import json
import logging
import re
from typing import List

from models.schemas import AnalysisReport, CVJDState, MatchResult, VerdictType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rubric-based GPT analysis prompt (enhanced from v1 prompt_tweaking.py)
# ---------------------------------------------------------------------------

REPORT_PROMPT = """You are a seasoned HR hiring manager at a large manufacturing and engineering conglomerate.
Perform a METICULOUS evaluation of the candidate's CV against the Job Description.

SCORING RUBRIC (use exactly these thresholds):
- 85–100: "Highly Suitable" — Strong alignment across skills, experience, education and responsibilities
- 70–84:  "Potentially Hireable" — Good fit with minor gaps; with some training could excel
- 50–69:  "Partially Suitable — Significant Gaps" — Overlaps in some areas but notable gaps in critical requirements
- 0–49:   "Not a Recommended Fit" — Insufficient alignment; major skill/experience gaps

Pre-computed scores (use as inputs, not your sole basis):
- Semantic similarity: {semantic_score}%
- Skill match: {skill_score}% ({matching_count}/{total_required} required skills)
- Experience: Candidate has {candidate_exp} yrs; JD requires {jd_min_exp} yrs

Return a VALID JSON object with EXACTLY these keys:
{{
    "confidence_score": <integer 0–100>,
    "final_verdict": "<one of the four verdicts above, verbatim>",
    "key_hireable_insights": [
        "Specific strength or alignment point 1",
        "Specific strength or alignment point 2",
        "Up to 5 items"
    ],
    "match_summary": "<2–3 sentence paragraph summarising fit, gaps, and recommendation>",
    "matching_skills": ["skill1", "skill2", ...],
    "missing_skills": ["skill1", "skill2", ...]
}}

Rules:
- confidence_score must be consistent with final_verdict (e.g. 72 → "Potentially Hireable")
- matching_skills: skills present in BOTH the CV and JD
- missing_skills: skills REQUIRED by JD but absent from the CV
- key_hireable_insights: concrete observations, NOT generic statements
- Return ONLY the JSON. No markdown, no preamble.

---
JOB DESCRIPTION:
{jd_text}

---
CANDIDATE CV:
{cv_text}

---
JSON Report:"""


# ---------------------------------------------------------------------------
# Agent node function
# ---------------------------------------------------------------------------

def generate_reports_node(state: CVJDState) -> CVJDState:
    """
    LangGraph node: generate_reports

    Reads:  state.candidate_matches, state.parsed_cv, state.cv_filename, state.cv_text
    Writes: state.final_reports, state.current_step

    Args:
        state: Current LangGraph CVJDState

    Returns:
        Updated CVJDState with final_reports populated
    """
    logger.info(
        "[generate_reports] Generating %d reports for: %s",
        len(state.candidate_matches), state.cv_filename
    )
    state.current_step = "generate_reports"

    if not state.candidate_matches:
        logger.warning("[generate_reports] No matches to generate reports for.")
        state.final_reports = []
        return state

    candidate_name = state.parsed_cv.candidate_name if state.parsed_cv else "Unknown"
    candidate_exp = state.parsed_cv.experience_years if state.parsed_cv else 0.0

    reports: List[AnalysisReport] = []
    for match in state.candidate_matches:
        try:
            report = _generate_single_report(
                cv_text=state.cv_text,
                cv_filename=state.cv_filename,
                candidate_name=candidate_name,
                candidate_exp=candidate_exp,
                match=match,
            )
            reports.append(report)
            logger.info(
                "[generate_reports] %s ↔ %s → %d%% (%s)",
                state.cv_filename, match.jd_filename,
                report.confidence_score, report.final_verdict
            )
        except Exception as e:
            logger.error(
                "[generate_reports] Failed for %s ↔ %s: %s",
                state.cv_filename, match.jd_filename, e
            )
            # Create a fallback report with scoring-based data
            reports.append(_fallback_report(
                cv_filename=state.cv_filename,
                candidate_name=candidate_name,
                match=match,
            ))

    state.final_reports = reports
    logger.info("[generate_reports] Done — %d reports generated.", len(reports))
    return state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_single_report(
    cv_text: str,
    cv_filename: str,
    candidate_name: str,
    candidate_exp: float,
    match: MatchResult,
) -> AnalysisReport:
    """Generate a single GPT analysis report for one CV-JD pair."""
    from config.settings import get_llm_client, get_model_name, settings

    client = get_llm_client()
    model = get_model_name()

    # Prepare prompt variables
    scoring = match.scoring
    semantic_pct = round(scoring.semantic_similarity * 100, 1)
    skill_pct = round(scoring.skill_match_ratio * 100, 1)
    n_matching = len(match.matching_skills)
    n_required = n_matching + len(match.missing_skills)
    jd_min_exp = _extract_exp_from_jd(match.jd_text)

    # Truncate texts for token safety (keep first 4000 chars each)
    cv_truncated = cv_text[:4000] if len(cv_text) > 4000 else cv_text
    jd_truncated = match.jd_text[:4000] if len(match.jd_text) > 4000 else match.jd_text

    prompt = REPORT_PROMPT.format(
        semantic_score=semantic_pct,
        skill_score=skill_pct,
        matching_count=n_matching,
        total_required=n_required if n_required > 0 else "N/A",
        candidate_exp=candidate_exp,
        jd_min_exp=jd_min_exp if jd_min_exp > 0 else "Not specified",
        jd_text=jd_truncated,
        cv_text=cv_truncated,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.gpt_temperature,
        max_tokens=settings.gpt_max_tokens,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    parsed = _safe_json_parse(raw)

    # Extract and validate verdict
    raw_verdict = parsed.get("final_verdict", "Not a Recommended Fit")
    verdict = _normalize_verdict(raw_verdict)

    return AnalysisReport(
        cv_filename=cv_filename,
        candidate_name=candidate_name,
        jd_filename=match.jd_filename,
        jd_title=match.jd_title,
        confidence_score=int(parsed.get("confidence_score", int(scoring.weighted_confidence_score))),
        scoring_breakdown=scoring,
        final_verdict=verdict,
        key_hireable_insights=_to_list(parsed.get("key_hireable_insights", [])),
        match_summary=parsed.get("match_summary", ""),
        matching_skills=_to_list(parsed.get("matching_skills", match.matching_skills)),
        missing_skills=_to_list(parsed.get("missing_skills", match.missing_skills)),
    )


def _fallback_report(
    cv_filename: str,
    candidate_name: str,
    match: MatchResult,
) -> AnalysisReport:
    """Create a report from scoring data when GPT call fails."""
    score = int(match.scoring.weighted_confidence_score)
    verdict = _score_to_verdict(score)
    return AnalysisReport(
        cv_filename=cv_filename,
        candidate_name=candidate_name,
        jd_filename=match.jd_filename,
        jd_title=match.jd_title,
        confidence_score=score,
        scoring_breakdown=match.scoring,
        final_verdict=verdict,
        key_hireable_insights=["Analysis generated from scoring metrics (GPT unavailable)"],
        match_summary=(
            f"Automated scoring indicates a {score}% match based on semantic similarity, "
            f"skill overlap, and experience alignment."
        ),
        matching_skills=match.matching_skills,
        missing_skills=match.missing_skills,
    )


def _safe_json_parse(raw: str) -> dict:
    """Parse JSON from LLM response with multiple fallback strategies."""
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

    logger.warning("[generate_reports] JSON parse failed. Raw: %s", raw[:200])
    return {}


def _normalize_verdict(raw: str) -> str:
    """Map raw LLM verdict string to one of the four valid VerdictType literals."""
    raw_lower = raw.lower()
    if "highly" in raw_lower:
        return "Highly Suitable"
    elif "potentially" in raw_lower or "hireable" in raw_lower:
        return "Potentially Hireable"
    elif "partially" in raw_lower or "significant" in raw_lower:
        return "Partially Suitable — Significant Gaps"
    else:
        return "Not a Recommended Fit"


def _score_to_verdict(score: int) -> str:
    """Convert numeric score to VerdictType using rubric thresholds."""
    if score >= 85:
        return "Highly Suitable"
    elif score >= 70:
        return "Potentially Hireable"
    elif score >= 50:
        return "Partially Suitable — Significant Gaps"
    else:
        return "Not a Recommended Fit"


def _extract_exp_from_jd(jd_text: str) -> float:
    """Extract minimum experience from JD text using regex."""
    if not jd_text:
        return 0.0
    patterns = [
        r"(\d+)\+?\s*(?:–|to|-)\s*\d+\s*years?",
        r"minimum\s+(\d+)\s*years?",
        r"at\s+least\s+(\d+)\s*years?",
        r"(\d+)\+\s*years?",
        r"(\d+)\s*years?\s+(?:of\s+)?experience",
    ]
    for pat in patterns:
        m = re.search(pat, jd_text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return 0.0


def _to_list(value) -> list:
    if isinstance(value, list):
        return [str(item).strip() for item in value if item]
    if isinstance(value, str) and value:
        return [value]
    return []
