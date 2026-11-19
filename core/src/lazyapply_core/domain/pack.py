"""EvidencePack: what retrieval hands to the matching step.

One entry per requirement, each with its own candidates. Searching per
requirement rather than once for the whole posting is what makes a gap
identifiable -- a requirement with nothing near it has no candidate evidence,
and that is a measurement rather than an opinion.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from lazyapply_core.domain.evidence import Evidence, ScoredEvidence
from lazyapply_core.domain.role import Requirement


class RequirementEvidence(BaseModel):
    """One requirement and the evidence retrieved for it."""

    model_config = ConfigDict(frozen=True)

    requirement: Requirement
    hits: tuple[ScoredEvidence, ...] = ()

    @property
    def best_score(self) -> float | None:
        """Highest similarity retrieved, or None if nothing was.

        Recorded for threshold tuning; never used directly to decide a
        verdict. Whether retrieved evidence actually satisfies a requirement
        is the model's call, not the distance's.
        """
        return self.hits[0].score if self.hits else None


class EvidencePack(BaseModel):
    model_config = ConfigDict(frozen=True)

    role_id: str
    items: tuple[RequirementEvidence, ...]

    def unique_evidence(self) -> tuple[Evidence, ...]:
        """Every distinct block in the pack, sorted by id.

        The same block is often retrieved for several requirements. The
        matching prompt carries each one once and refers to it by id, so a
        popular block does not consume the context five times over.
        """
        seen = {hit.evidence.evidence_id: hit.evidence for item in self.items for hit in item.hits}
        return tuple(seen[key] for key in sorted(seen))
