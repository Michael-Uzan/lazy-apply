"""What the matching model is asked to return.

Separate from the domain types for two reasons. The model supplies only what
it can judge -- a status, citations and a sentence -- while coverage, score,
verdict and gaps are computed from those. And ``evidence_ids`` is a plain
``str`` here rather than the domain's pattern-constrained ``EvidenceId``, so a
hallucinated citation fails in the verifier with a message naming the id,
instead of failing as a schema violation somewhere less legible.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from lazyapply_core.domain import MatchStatus


class JudgedRequirement(BaseModel):
    requirement_id: str = Field(description="The requirement id being judged, copied exactly.")

    status: MatchStatus = Field(
        description=(
            "STRONG if the evidence clearly demonstrates the requirement, "
            "PARTIAL if it is adjacent or weaker than asked, "
            "NONE if nothing shown supports it."
        )
    )

    evidence_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Ids of the evidence blocks that justify this status, copied "
            "exactly from the catalogue. Required for STRONG and PARTIAL. "
            "Empty for NONE."
        ),
    )

    rationale: str = Field(description="One sentence, for the candidate to read.")


class Judgment(BaseModel):
    matches: list[JudgedRequirement] = Field(
        description="Exactly one entry per requirement given, in the same order."
    )
