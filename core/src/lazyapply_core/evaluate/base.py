"""The matching seam."""

from __future__ import annotations

from typing import Protocol

from lazyapply_core.domain import EvidencePack, Profile
from lazyapply_core.evaluate.schema import Judgment


class EvaluationError(Exception):
    """Matching failed, or returned something unusable.

    Covers provider faults -- rate limits, outages, refusals -- and results
    that validate as JSON but cannot be used. Grounding failures are separate:
    those are ``verifier.CitationError``.
    """


class Evaluator(Protocol):
    def judge(self, pack: EvidencePack, profile: Profile) -> Judgment:
        """Judge every requirement in ``pack`` against the evidence and facts."""
