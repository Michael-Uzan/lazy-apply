"""Evidence: one self-contained, citable claim about the user's career.

One ``## block`` in an ``evidence/*.md`` file becomes exactly one Evidence,
which becomes exactly one vector in Qdrant. Chunks are never split mid-story —
that is the entire reason the evidence files are hand-authored.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

#: Calendar month, e.g. "2023-05". Day-level precision is noise for a CV.
YearMonth = Annotated[str, StringConstraints(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]

#: An end date is either a month or the open-ended present.
EndDate = YearMonth | Literal["present"]

#: Floating-point slack on cosine scores. A unit vector dotted with itself
#: computes to 1.0000000000000002 — noise, not an out-of-range score.
_SCORE_TOLERANCE = 1e-9

#: Readable citation anchor, e.g. "ev_leverate__multi-agent-llm-orchestration".
_SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"
EvidenceId = Annotated[str, StringConstraints(pattern=rf"^ev_{_SLUG}__{_SLUG}$")]


class Evidence(BaseModel):
    """A single retrievable, citable unit of career evidence."""

    model_config = ConfigDict(frozen=True)

    evidence_id: EvidenceId
    """Readable, stable across rewording. Cited verbatim in a FitReport."""

    title: str
    """The ``##`` heading, e.g. "Multi-agent LLM orchestration"."""

    body: str
    """Raw prose, shown to humans. Never the thing that gets embedded."""

    composed_text: str
    """What actually gets embedded: company + title + tags + body.

    Requirement queries are terse ("5+ years React"). Folding the company and
    tags into the embedded text measurably improves matching against them.
    """

    content_hash: str
    """Digest of ``composed_text``. Drives the embedding cache, not identity."""

    company: str | None = None
    role: str | None = None
    start: YearMonth | None = None
    end: EndDate | None = None

    tags: tuple[str, ...] = ()
    source_file: str
    """Relative path of the markdown file this block came from."""


class ScoredEvidence(BaseModel):
    """Evidence plus the similarity score that retrieved it."""

    model_config = ConfigDict(frozen=True)

    evidence: Evidence
    score: float = Field(ge=-1.0, le=1.0)
    """Cosine similarity. Recorded for eval tuning, never used directly to
    decide a verdict."""

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_rounding_noise(cls, value: float) -> float:
        """Clamp scores a hair outside [-1, 1]; leave real violations to fail.

        Clamping everything would hide genuine bugs — a score of 5.0 means
        something upstream is broken and should say so.
        """
        if 1.0 < value <= 1.0 + _SCORE_TOLERANCE:
            return 1.0
        if -1.0 - _SCORE_TOLERANCE <= value < -1.0:
            return -1.0
        return value
