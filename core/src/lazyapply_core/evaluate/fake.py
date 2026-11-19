"""Evaluators for tests and for running the CLI offline.

``FakeEvaluator`` returns whatever it was given and counts calls, so a test
can prove one request was made rather than ten.

``ScoreBasedEvaluator`` judges from retrieval scores alone. It exists so the
whole ``fit`` command can be exercised with no key and no cost -- it is
explicitly *not* the design. Deciding sufficiency from a cosine distance is
the approach this project rejected after measuring it: across two real
postings the matching and the non-matching score ranges were separated by
0.020, with a correct match at 0.353 and a nonsensical one at 0.333. Judging
on that number would be a coin flip dressed up as a threshold.
"""

from __future__ import annotations

from lazyapply_core.domain import (
    EvidencePack,
    MatchStatus,
    Profile,
    RequirementEvidence,
)
from lazyapply_core.evaluate.schema import JudgedRequirement, Judgment

STRONG_AT = 0.45
"""Only meaningful for the fake. Real sufficiency is the model's call."""


class FakeEvaluator:
    def __init__(self, result: Judgment, *, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.calls = 0

    def judge(self, pack: EvidencePack, profile: Profile) -> Judgment:
        self.calls += 1
        if self._error:
            raise self._error
        return self._result


class ScoreBasedEvaluator:
    """Judges by retrieval score. Plumbing only -- see the module docstring."""

    def __init__(self, *, strong_at: float = STRONG_AT) -> None:
        self._strong_at = strong_at
        self.calls = 0

    def judge(self, pack: EvidencePack, profile: Profile) -> Judgment:
        self.calls += 1
        return Judgment(matches=[self._judge_one(item) for item in pack.items])

    def _judge_one(self, item: RequirementEvidence) -> JudgedRequirement:
        requirement, hits = item.requirement, item.hits

        if not hits:
            return JudgedRequirement(
                requirement_id=requirement.id,
                status=MatchStatus.NONE,
                rationale="Nothing retrieved for this requirement.",
            )

        best = hits[0]
        status = MatchStatus.STRONG if best.score >= self._strong_at else MatchStatus.PARTIAL
        return JudgedRequirement(
            requirement_id=requirement.id,
            status=status,
            evidence_ids=[best.evidence.evidence_id],
            rationale=f"Nearest evidence scored {best.score:.3f} (fake evaluator).",
        )
