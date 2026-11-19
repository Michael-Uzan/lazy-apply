"""Role: a job description, decomposed into checkable requirements.

The decomposition is the highest-leverage step in the pipeline. Getting
MUST_HAVE vs NICE_TO_HAVE wrong is how a missing "Kubernetes a plus" sinks an
otherwise strong candidate — so this is the one call that runs on the expensive
model, and it is cached forever by ``role_id``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class RequirementKind(StrEnum):
    MUST_HAVE = "must_have"
    NICE_TO_HAVE = "nice_to_have"

    @property
    def label(self) -> str:
        """How this reads to a person: "must-have", "nice-to-have".

        Defined once, on the type. Four modules used to format it themselves
        and one of them compared the result back against a literal, which
        would have broken silently the moment any of them was reworded.
        """
        return self.value.replace("_", "-")


class RequirementCategory(StrEnum):
    SKILL = "skill"
    EXPERIENCE = "experience"
    DOMAIN = "domain"
    EDUCATION = "education"
    OTHER = "other"


class Requirement(BaseModel):
    """One checkable claim the role makes about its candidate."""

    model_config = ConfigDict(frozen=True)

    id: str
    """Short slug, unique within the role, e.g. "req_vector_databases"."""

    text: str
    """Restated as a self-contained requirement, not copied verbatim from the
    posting. This text becomes the retrieval query, so it must read like
    something that could match an impact story."""

    kind: RequirementKind
    category: RequirementCategory

    @property
    def is_hard_fact(self) -> bool:
        """True when this requirement is answered by profile.json, not by
        retrieval.

        A degree is a fact the candidate either holds or does not. Searching
        impact stories for it cannot answer the question, but retrieval always
        returns its nearest neighbours anyway -- in testing, "bachelor's in
        electrical engineering" pulled back a story about mentoring culture at
        0.251 and gave the matching step something irrelevant to reason over.
        These requirements therefore skip retrieval entirely and are judged
        against the verbatim facts.
        """
        return self.category is RequirementCategory.EDUCATION


class Role(BaseModel):
    model_config = ConfigDict(frozen=True)

    role_id: str
    """Content hash of ``raw_text``. The requirement-extraction cache key."""

    title: str
    company: str | None = None
    raw_text: str
    requirements: tuple[Requirement, ...] = ()

    @property
    def must_haves(self) -> tuple[Requirement, ...]:
        return tuple(r for r in self.requirements if r.kind is RequirementKind.MUST_HAVE)

    @property
    def nice_to_haves(self) -> tuple[Requirement, ...]:
        return tuple(r for r in self.requirements if r.kind is RequirementKind.NICE_TO_HAVE)
