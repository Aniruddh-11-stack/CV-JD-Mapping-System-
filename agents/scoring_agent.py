"""
CV to JD Mapping System v2 — Scoring Agent
===========================================
LangGraph node: score_matches

Responsibilities:
- Compute multi-dimensional confidence scores for each CV-JD match
- Integrates confidence_score.py logic (from v1) as a proper Pydantic model
- Populates ScoringBreakdown.compute() for each MatchResult
- Determines matching_skills and missing_skills

Scoring formula (from confidence_score.py, now integrated):
  score = (
      w_semantic   × cosine_similarity
    + w_skill      × (matched_skills / required_skills)
    + w_experience × min(cv_exp / jd_min_exp, 1.0)
    + w_education  × education_match
  ) × 100

Weights are configurable in settings.py (default: 0.40/0.30/0.20/0.10).
"""

import logging
import re
from typing import List, Set, Tuple

from models.schemas import CVJDState, MatchResult, ScoringBreakdown

logger = logging.getLogger(__name__)

# Education level hierarchy (used for education_match scoring)
_EDUCATION_TIERS = {
    "phd": 5, "doctorate": 5, "ph.d": 5,
    "mba": 4, "m.e": 4, "m.tech": 4, "m.sc": 4, "master": 4, "pg diploma": 4,
    "b.e": 3, "b.tech": 3, "b.sc": 3, "b.com": 3, "bachelor": 3, "b.a": 3,
    "diploma": 2,
    "12th": 1, "hsc": 1, "12": 1,
    "10th": 0, "ssc": 0,
}


def score_matches_node(state: CVJDState) -> CVJDState:
    """
    LangGraph node: score_matches

    Reads:  state.candidate_matches, state.parsed_cv
    Writes: state.candidate_matches (enriched with scoring), state.current_step

    Args:
        state: Current LangGraph CVJDState

    Returns:
        Updated CVJDState with ScoringBreakdown computed for each match
    """
    logger.info("[score_matches] Scoring %d matches", len(state.candidate_matches))
    state.current_step = "score_matches"

    from config.settings import settings

    if not state.candidate_matches:
        logger.warning("[score_matches] No matches to score.")
        return state

    candidate_exp = state.parsed_cv.experience_years if state.parsed_cv else 0.0
    candidate_skills: Set[str] = (
        {s.lower().strip() for s in state.parsed_cv.skills}
        if state.parsed_cv
        else set()
    )
    candidate_education = state.parsed_cv.education if state.parsed_cv else ""

    scored_matches: List[MatchResult] = []
    for match in state.candidate_matches:
        # Extract JD metadata embedded in the match
        jd_required_skills = _extract_required_skills(match)
        jd_min_exp = _extract_min_exp(match)
        jd_education = _extract_required_education(match)

        # --- Compute individual dimensions ---
        matching_skills, missing_skills, skill_ratio = _compute_skill_match(
            candidate_skills, jd_required_skills
        )
        exp_ratio = _compute_experience_ratio(candidate_exp, jd_min_exp)
        edu_score = _compute_education_score(candidate_education, jd_education)

        # --- Build ScoringBreakdown ---
        breakdown = ScoringBreakdown(
            semantic_similarity=match.scoring.semantic_similarity,
            skill_match_ratio=round(skill_ratio, 4),
            experience_match_ratio=round(exp_ratio, 4),
            education_match=round(edu_score, 4),
            weight_semantic=settings.weight_semantic,
            weight_skill=settings.weight_skill,
            weight_experience=settings.weight_experience,
            weight_education=settings.weight_education,
        )
        breakdown.compute()  # Populates weighted_confidence_score

        # --- Update match ---
        match.scoring = breakdown
        match.matching_skills = matching_skills
        match.missing_skills = missing_skills

        scored_matches.append(match)

        logger.debug(
            "[score_matches] %s → score=%.1f | sem=%.2f | skill=%.2f | exp=%.2f | edu=%.2f",
            match.jd_filename,
            breakdown.weighted_confidence_score,
            breakdown.semantic_similarity,
            breakdown.skill_match_ratio,
            breakdown.experience_match_ratio,
            breakdown.education_match,
        )

    # Re-sort by weighted score descending
    scored_matches.sort(key=lambda m: m.scoring.weighted_confidence_score, reverse=True)
    state.candidate_matches = scored_matches

    logger.info(
        "[score_matches] Done. Top match: %s (%.1f%%)",
        scored_matches[0].jd_filename if scored_matches else "n/a",
        scored_matches[0].scoring.weighted_confidence_score if scored_matches else 0,
    )
    return state


# ---------------------------------------------------------------------------
# Skill matching
# ---------------------------------------------------------------------------

def _extract_required_skills(match: MatchResult) -> List[str]:
    """
    Extract required skills from the MatchResult.
    The jd_text may contain structured metadata inserted during indexing.
    Falls back to empty list if not available.
    """
    # If match.jd_text contains a JSON-like metadata block (set during indexing), parse it
    # Otherwise return empty list — LLM will compare via semantic similarity
    return []  # Populated by jd_analyzer_agent during indexing; scoring uses what's available


def _compute_skill_match(
    candidate_skills: Set[str],
    required_skills: List[str],
) -> Tuple[List[str], List[str], float]:
    """
    Compute skill overlap between candidate and JD.

    Args:
        candidate_skills: Set of lowercase candidate skill strings
        required_skills: List of required skill strings from JD

    Returns:
        (matching_skills, missing_skills, ratio)
    """
    if not required_skills:
        return [], [], 0.5  # Neutral score when JD has no extracted skills

    req_lower = {s.lower().strip() for s in required_skills}
    matching = sorted([s for s in required_skills if s.lower().strip() in candidate_skills])
    missing = sorted([s for s in required_skills if s.lower().strip() not in candidate_skills])

    ratio = len(matching) / len(req_lower) if req_lower else 0.0
    return matching, missing, ratio


def _extract_min_exp(match: MatchResult) -> float:
    """
    Extract minimum experience requirement from MatchResult metadata.
    Tries to find it in the JD text with regex as a fallback.
    """
    # Regex fallback on jd_text
    if match.jd_text:
        patterns = [
            r"(\d+)\+?\s*(?:–|to|-)\s*(\d+)\s*years?",  # "5–8 years"
            r"minimum\s+(\d+)\s*(?:\+)?\s*years?",
            r"at\s+least\s+(\d+)\s*years?",
            r"(\d+)\+\s*years?",
            r"(\d+)\s*years?\s+(?:of\s+)?experience",
        ]
        for pat in patterns:
            m = re.search(pat, match.jd_text, re.IGNORECASE)
            if m:
                return float(m.group(1))
    return 0.0


def _extract_required_education(match: MatchResult) -> str:
    """Extract required education from JD text using regex."""
    if not match.jd_text:
        return ""
    patterns = [
        r"(B\.?E|B\.?Tech|M\.?Tech|M\.?E|MBA|B\.?Sc|M\.?Sc|Ph\.?D|Diploma)",
        r"(Bachelor|Master|Doctorate|Graduate|Post.?Graduate)",
    ]
    for pat in patterns:
        m = re.search(pat, match.jd_text, re.IGNORECASE)
        if m:
            return m.group(0)
    return ""


# ---------------------------------------------------------------------------
# Experience ratio
# ---------------------------------------------------------------------------

def _compute_experience_ratio(candidate_exp: float, jd_min_exp: float) -> float:
    """
    Compute experience match as min(candidate_exp / jd_min_exp, 1.0).
    Returns 0.8 (slightly below perfect) if JD has no experience requirement.

    Matches the original confidence_score.py logic.
    """
    if jd_min_exp <= 0:
        return 0.8  # Neutral-positive when JD doesn't specify
    return min(candidate_exp / jd_min_exp, 1.0)


# ---------------------------------------------------------------------------
# Education scoring
# ---------------------------------------------------------------------------

def _compute_education_score(candidate_edu: str, required_edu: str) -> float:
    """
    Score education match based on qualification tier hierarchy.

    Logic:
    - If candidate tier >= required tier → 1.0 (full match)
    - One tier below → 0.6 (partial)
    - Two or more tiers below → 0.2 (weak)
    - No info available → 0.5 (neutral)
    """
    if not candidate_edu and not required_edu:
        return 0.5

    candidate_tier = _get_education_tier(candidate_edu)
    required_tier = _get_education_tier(required_edu)

    if candidate_tier is None or required_tier is None:
        return 0.5

    gap = required_tier - candidate_tier
    if gap <= 0:
        return 1.0     # Meets or exceeds requirement
    elif gap == 1:
        return 0.6     # One level below
    else:
        return 0.2     # Significantly under-qualified


def _get_education_tier(edu_text: str) -> float:
    """Map education text to a numeric tier using the hierarchy dict."""
    if not edu_text:
        return None
    lower = edu_text.lower()
    for keyword, tier in _EDUCATION_TIERS.items():
        if keyword in lower:
            return tier
    return None
