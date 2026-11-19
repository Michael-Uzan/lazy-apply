"""What the model is asked to return.

Deliberately separate from the domain types. The model supplies the parts it
can judge -- the title and the requirements -- while role_id and raw_text are
computed by us. Letting a model produce its own cache key would be absurd,
and the separation keeps the wire format free to change without touching the
domain.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from lazyapply_core.domain import RequirementCategory, RequirementKind


class ExtractedRequirement(BaseModel):
    id: str = Field(description="Short snake_case slug, unique within the role.")

    text: str = Field(
        description=(
            "The requirement restated as something a candidate's impact story "
            "could match. Not copied from the posting."
        )
    )

    kind: RequirementKind = Field(
        description="MUST_HAVE only if the posting treats it as required."
    )

    category: RequirementCategory


class ExtractedRole(BaseModel):
    title: str
    company: str | None = None
    requirements: list[ExtractedRequirement]
