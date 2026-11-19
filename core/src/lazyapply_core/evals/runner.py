"""Scoring a pack and a report against the golden labels.

Both functions are pure: they take what the pipeline already produced plus
the labels, and return numbers. No embedder, no store, no model -- so the
scoring logic is testable without spending anything, and the expensive part
(building the pack, running the judgment) happens once in the CLI.

Label drift is treated as an error rather than skipped. Re-extracting a
posting can rename a requirement, and silently dropping the stale label would
shrink the measured set -- which reads as an improvement.
"""

from __future__ import annotations

from dataclasses import dataclass

from lazyapply_core.config import Thresholds
from lazyapply_core.domain import (
    EvidencePack,
    FitReport,
    MatchStatus,
    RequirementMatch,
    Role,
    Verdict,
)
from lazyapply_core.evals.golden import GoldenCase
from lazyapply_core.evals.metrics import Agreement, agreement, mean, recall_at_k, reciprocal_rank
from lazyapply_core.evaluate.verdict import build_report


class LabelDriftError(Exception):
    """The golden labels and the extracted requirements no longer line up."""


@dataclass(frozen=True)
class RequirementDetail:
    """One requirement's retrieval outcome, for drilling into a bad average."""

    requirement_id: str
    rank: int | None
    """Position of the first relevant block, or None if none was retrieved."""

    recall: float
    retrieved: tuple[str, ...]
    relevant: frozenset[str]

    @property
    def is_perfect(self) -> bool:
        """Found every relevant block, with one of them first.

        Both halves matter: rank 1 with half the relevant blocks missing is
        still a result worth looking at, and a full-recall hit buried at rank
        3 means the model reads two irrelevant blocks before the right one.
        """
        return self.rank == 1 and self.recall == 1.0

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(sorted(self.relevant - set(self.retrieved)))


@dataclass(frozen=True)
class RetrievalScore:
    role: str
    k: int
    """The cutoff these numbers were computed at."""

    scored: int
    """Requirements with at least one relevant block to find."""

    unscoreable: int
    """Requirements with no correct answer. Excluded, not failed."""

    recall: float | None
    mrr: float | None
    details: tuple[RequirementDetail, ...]


@dataclass(frozen=True)
class MatchingScore:
    role: str
    agreement: Agreement
    expected_verdict: Verdict
    actual_verdict: Verdict

    @property
    def verdict_matched(self) -> bool:
        return self.expected_verdict is self.actual_verdict


def score_retrieval(case: GoldenCase, pack: EvidencePack, *, k: int) -> RetrievalScore:
    """Measure recall@k and MRR for one posting.

    Raises:
        LabelDriftError: if the labels and the pack disagree about which
            requirements exist.
    """
    hits = {
        item.requirement.id: tuple(h.evidence.evidence_id for h in item.hits) for item in pack.items
    }
    _check_drift({r.id for r in case.requirements}, set(hits), case.role)

    details = []
    recalls: list[float] = []
    ranks: list[float] = []

    for requirement in case.scoreable:
        retrieved = hits[requirement.id]
        relevant = set(requirement.relevant)

        value = recall_at_k(retrieved, relevant=relevant, k=k)
        rank = reciprocal_rank(retrieved[:k], relevant=relevant)

        recalls.append(value)
        ranks.append(rank)
        details.append(
            RequirementDetail(
                requirement_id=requirement.id,
                rank=round(1 / rank) if rank else None,
                recall=value,
                retrieved=retrieved,
                relevant=requirement.relevant,
            )
        )

    return RetrievalScore(
        role=case.role,
        k=k,
        scored=len(recalls),
        unscoreable=len(case.requirements) - len(recalls),
        recall=mean(recalls),
        mrr=mean(ranks),
        details=tuple(details),
    )


def score_matching(case: GoldenCase, report: FitReport) -> MatchingScore:
    """Compare the model's statuses and the computed verdict against a person's.

    Raises:
        LabelDriftError: if the labels and the report disagree about which
            requirements exist.
    """
    actual: dict[str, MatchStatus] = {m.requirement_id: m.status for m in report.matches}
    _check_drift({r.id for r in case.requirements}, set(actual), case.role)

    pairs = [(r.status, actual[r.id]) for r in case.requirements]

    return MatchingScore(
        role=case.role,
        agreement=agreement(pairs),
        expected_verdict=case.expected_verdict,
        actual_verdict=report.verdict,
    )


@dataclass(frozen=True)
class ThresholdMargin:
    """Where one posting lands relative to the verdict boundaries.

    Computed from the *human* labels, not the model's, so it measures the
    thresholds rather than the model. A posting at 1.0 coverage confirms
    nothing about whether 0.80 is the right cutoff -- only a posting near a
    boundary does that, and ``distance_to_boundary`` is how much each one
    says.
    """

    role: str
    coverage: float
    score: int
    verdict: Verdict
    distance_to_boundary: float


def score_thresholds(case: GoldenCase, role: Role, thresholds: Thresholds) -> ThresholdMargin:
    """Derive the verdict a person's labels imply, and its margin."""
    matches = tuple(
        RequirementMatch(requirement_id=r.id, status=r.status, rationale="hand-labelled")
        for r in case.requirements
    )
    report = build_report(role, matches, profile_version="golden", thresholds=thresholds)

    return ThresholdMargin(
        role=case.role,
        coverage=report.must_have_coverage,
        score=report.score,
        verdict=report.verdict,
        distance_to_boundary=min(
            abs(report.must_have_coverage - thresholds.skip_below),
            abs(report.must_have_coverage - thresholds.generate_at),
        ),
    )


def _check_drift(labelled: set[str], present: set[str], role: str) -> None:
    renamed = labelled - present
    if renamed:
        raise LabelDriftError(
            f"{role}: labelled but no longer extracted (renamed or removed): "
            f"{', '.join(sorted(renamed))}"
        )

    unlabelled = present - labelled
    if unlabelled:
        raise LabelDriftError(f"{role}: unlabelled requirements: {', '.join(sorted(unlabelled))}")
