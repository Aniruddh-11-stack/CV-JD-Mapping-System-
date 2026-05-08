"""
CV to JD Mapping System v2 — PydanticAI Type-Safe Schemas
=========================================================
All data models used across agents, graph, API, and UI.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from datetime import datetime


# ---------------------------------------------------------------------------
# CV Models
# ---------------------------------------------------------------------------

class ParsedCV(BaseModel):
    """Structured data extracted from a candidate's CV by the CV Parser Agent."""

    candidate_name: str = Field(default="Unknown", description="Full name of the candidate")
    candidate_title: str = Field(default="", description="Current/desired job title")
    skills: List[str] = Field(default_factory=list, description="Technical and soft skills list")
    experience_years: float = Field(default=0.0, ge=0, description="Total years of experience")
    education: str = Field(default="", description="Highest educational qualification")
    work_history: List[str] = Field(default_factory=list, description="Previous roles/companies")
    certifications: List[str] = Field(default_factory=list, description="Professional certifications")
    department: str = Field(default="", description="Inferred department/domain")
    raw_text: str = Field(default="", description="Original CV text")
    enriched_text: str = Field(default="", description="Enriched CV text with structured metadata")

    @field_validator("experience_years", mode="before")
    @classmethod
    def parse_experience(cls, v):
        if isinstance(v, str):
            import re
            match = re.search(r"\d+\.?\d*", v)
            return float(match.group()) if match else 0.0
        return float(v) if v else 0.0


# ---------------------------------------------------------------------------
# JD Models
# ---------------------------------------------------------------------------

class ParsedJD(BaseModel):
    """Structured data extracted from a Job Description by the JD Analyzer Agent."""

    job_title: str = Field(default="", description="Job title")
    required_skills: List[str] = Field(default_factory=list, description="Must-have skills")
    nice_to_have_skills: List[str] = Field(default_factory=list, description="Preferred skills")
    min_experience_years: float = Field(default=0.0, ge=0, description="Minimum experience required")
    required_education: str = Field(default="", description="Required educational qualification")
    department: str = Field(default="", description="Department/function")
    location: str = Field(default="", description="Job location")
    key_responsibilities: List[str] = Field(default_factory=list, description="Key job responsibilities")
    raw_text: str = Field(default="", description="Original JD text")
    filename: str = Field(default="", description="Source JD filename")

    @field_validator("min_experience_years", mode="before")
    @classmethod
    def parse_experience(cls, v):
        if isinstance(v, str):
            import re
            match = re.search(r"\d+\.?\d*", v)
            return float(match.group()) if match else 0.0
        return float(v) if v else 0.0


# ---------------------------------------------------------------------------
# Scoring Models
# ---------------------------------------------------------------------------

class ScoringBreakdown(BaseModel):
    """Detailed multi-dimensional scoring breakdown for a CV-JD pair."""

    semantic_similarity: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Cosine similarity between CV and JD embeddings (0–1)"
    )
    skill_match_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of required JD skills found in CV (0–1)"
    )
    experience_match_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="min(candidate_exp / required_exp, 1.0) — capped at 1 (0–1)"
    )
    education_match: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Education alignment score (0–1)"
    )
    weighted_confidence_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Final composite confidence score (0–100)"
    )

    # Weights used (stored for transparency)
    weight_semantic: float = Field(default=0.40)
    weight_skill: float = Field(default=0.30)
    weight_experience: float = Field(default=0.20)
    weight_education: float = Field(default=0.10)

    def compute(self) -> float:
        """Recompute weighted score from components."""
        score = (
            self.weight_semantic * self.semantic_similarity
            + self.weight_skill * self.skill_match_ratio
            + self.weight_experience * self.experience_match_ratio
            + self.weight_education * self.education_match
        ) * 100
        self.weighted_confidence_score = round(score, 2)
        return self.weighted_confidence_score


# ---------------------------------------------------------------------------
# Match & Report Models
# ---------------------------------------------------------------------------

VerdictType = Literal[
    "Highly Suitable",
    "Potentially Hireable",
    "Partially Suitable — Significant Gaps",
    "Not a Recommended Fit",
]


class MatchResult(BaseModel):
    """Intermediate result after retrieval + scoring (before GPT report)."""

    jd_filename: str = Field(description="Source JD filename")
    jd_title: str = Field(default="", description="Job title from parsed JD")
    similarity_score: float = Field(default=0.0, description="Raw FAISS cosine similarity")
    scoring: ScoringBreakdown = Field(default_factory=ScoringBreakdown)
    matching_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    jd_text: str = Field(default="", description="Full JD text for GPT analysis")


class AnalysisReport(BaseModel):
    """Final analysis report generated by the Report Agent (GPT-powered)."""

    cv_filename: str = Field(description="Source CV filename")
    candidate_name: str = Field(default="Unknown")
    jd_filename: str = Field(description="Matched JD filename")
    jd_title: str = Field(default="")

    # Scores
    confidence_score: int = Field(
        default=0, ge=0, le=100,
        description="Overall match confidence percentage"
    )
    scoring_breakdown: Optional[ScoringBreakdown] = Field(
        default=None,
        description="Detailed scoring breakdown (semantic + skill + experience + education)"
    )

    # Verdict
    final_verdict: str = Field(default="Not a Recommended Fit")
    key_hireable_insights: List[str] = Field(default_factory=list)
    match_summary: str = Field(default="")

    # Skills
    matching_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)

    # Metadata
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def verdict_color(self) -> str:
        """Return color for UI verdict badge."""
        score = self.confidence_score
        if score >= 85:
            return "#28a745"   # green
        elif score >= 70:
            return "#17a2b8"   # blue
        elif score >= 50:
            return "#ffc107"   # amber
        else:
            return "#dc3545"   # red

    def to_flat_dict(self) -> dict:
        """Flatten to a dict suitable for DataFrame/Excel export."""
        return {
            "CV_Filename": self.cv_filename,
            "Candidate_Name": self.candidate_name,
            "JD_Filename": self.jd_filename,
            "JD_Title": self.jd_title,
            "Confidence_Score": self.confidence_score,
            "Final_Verdict": self.final_verdict,
            "Match_Summary": self.match_summary,
            "Key_Hireable_Insights": "; ".join(self.key_hireable_insights),
            "Matching_Skills": ", ".join(self.matching_skills),
            "Missing_Skills": ", ".join(self.missing_skills),
            "Semantic_Similarity": round(self.scoring_breakdown.semantic_similarity * 100, 1) if self.scoring_breakdown else "",
            "Skill_Match_%": round(self.scoring_breakdown.skill_match_ratio * 100, 1) if self.scoring_breakdown else "",
            "Experience_Match_%": round(self.scoring_breakdown.experience_match_ratio * 100, 1) if self.scoring_breakdown else "",
            "Timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class CVJDState(BaseModel):
    """
    LangGraph state flowing through the agent graph.
    Each node reads from and writes back to this state.
    """

    # --- Inputs ---
    cv_text: str = Field(default="")
    cv_filename: str = Field(default="")

    # JD pool (indexed)
    jd_pool: List[dict] = Field(
        default_factory=list,
        description="List of dicts with 'filename', 'text', 'embedding' for each indexed JD"
    )

    # --- Agent Outputs ---
    parsed_cv: Optional[ParsedCV] = Field(default=None)
    candidate_matches: List[MatchResult] = Field(default_factory=list)
    final_reports: List[AnalysisReport] = Field(default_factory=list)

    # --- Control ---
    error: Optional[str] = Field(default=None)
    current_step: str = Field(default="init")
    top_k: int = Field(default=3, description="Number of top JD matches to return")

    class Config:
        arbitrary_types_allowed = True
