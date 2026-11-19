"""The fit pipeline: judge, verify, decide.

Three steps, in this order, and the order is the design:

1. The model judges each requirement -- status, citations, one sentence.
2. The verifier checks the grounding. A judgment citing evidence that was
   never sent is a hard failure here.
3. Pure code computes coverage, score, verdict and gaps.

An ungrounded judgment must never reach step 3, because the verdict is what
the user acts on. Failing loudly beats a convincing report built on an
invented citation.
"""

from __future__ import annotations

from lazyapply_core.config import Thresholds
from lazyapply_core.domain import (
    EvidencePack,
    FitReport,
    Profile,
    RequirementMatch,
    Role,
)
from lazyapply_core.evaluate.base import Evaluator
from lazyapply_core.evaluate.schema import Judgment
from lazyapply_core.evaluate.verdict import build_report
from lazyapply_core.evaluate.verifier import verify


def evaluate_fit(
    role: Role,
    pack: EvidencePack,
    profile: Profile,
    *,
    evaluator: Evaluator,
    thresholds: Thresholds,
    profile_version: str,
) -> FitReport:
    """Judge ``pack`` against ``profile`` and return a grounded FitReport.

    Raises:
        EvaluationError: the model call failed or returned nothing usable.
        CitationError: a judgment is not grounded in the evidence sent.
    """
    judgment = evaluator.judge(pack, profile)
    verify(judgment, pack)

    return build_report(
        role,
        _to_matches(judgment, pack),
        profile_version=profile_version,
        thresholds=thresholds,
    )


def _to_matches(judgment: Judgment, pack: EvidencePack) -> tuple[RequirementMatch, ...]:
    """Wire schema to domain, attaching each requirement's retrieval score.

    The score comes from the pack, never from the response -- a model has no
    way to know a cosine distance, and asking it to echo one back would be an
    invitation to invent a number.
    """
    scores = {item.requirement.id: item.best_score for item in pack.items}

    return tuple(
        RequirementMatch(
            requirement_id=match.requirement_id,
            status=match.status,
            evidence_ids=tuple(match.evidence_ids),
            best_retrieval_score=scores.get(match.requirement_id),
            rationale=match.rationale,
        )
        for match in judgment.matches
    )
