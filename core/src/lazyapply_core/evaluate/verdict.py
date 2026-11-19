"""Coverage, score and verdict -- the decision, as a pure function.

The model never emits a verdict. It reports a status per requirement; this
module turns those statuses into a decision. The consequences are the reason
the system is built this way:

- Reproducible. The same statuses always give the same verdict.
- Testable with no API key and no network, which is why this is the most
  heavily tested module in the project.
- Tunable. The thresholds live in ``config.Thresholds`` and get swept against
  the golden set in slice 9, without touching a prompt.
- Explainable. "3 of 7 must-haves unmet" is a sentence a user can argue with.
"""

from __future__ import annotations

from datetime import UTC, datetime

from lazyapply_core.config import Thresholds
from lazyapply_core.domain import (
    FitReport,
    MatchStatus,
    Requirement,
    RequirementKind,
    RequirementMatch,
    Role,
    Verdict,
)

_WEIGHT = {MatchStatus.STRONG: 1.0, MatchStatus.PARTIAL: 0.5, MatchStatus.NONE: 0.0}
"""A partial counts as half. Zero would reject candidates who can argue the
point in an interview; one would send them in unprepared."""

_MUST_HAVE_WEIGHT = 0.8
"""Must-haves carry most of the score. Nice-to-haves only separate two
candidates who have both already cleared the hard requirements."""

_UNJUDGED = "Not judged by the model; counted as unmet."


def must_have_coverage(matches: tuple[RequirementMatch, ...], role: Role) -> float:
    """Fraction of hard requirements met, partials counting half.

    Denominator is every must-have in the role, not every must-have the model
    happened to answer -- otherwise skipping a requirement would raise the
    score instead of lowering it.
    """
    return _weighted(matches, role.must_haves)


def fit_score(matches: tuple[RequirementMatch, ...], role: Role) -> int:
    """0-100, must-haves weighted at 80%.

    Distinct from coverage: coverage answers "may I apply at all", score
    ranks two applications that both may.
    """
    must = _weighted(matches, role.must_haves)
    nice = role.nice_to_haves
    if not nice:
        return round(100 * must)
    weighted = _MUST_HAVE_WEIGHT * must + (1 - _MUST_HAVE_WEIGHT) * _weighted(matches, nice)
    return round(100 * weighted)


def verdict_for(coverage: float, score: int, thresholds: Thresholds) -> Verdict:
    """The decision. Three lines, deliberately."""
    if coverage < thresholds.skip_below:
        return Verdict.SKIP
    if coverage >= thresholds.generate_at and score >= thresholds.min_score:
        return Verdict.GENERATE
    return Verdict.BORDERLINE


def build_report(
    role: Role,
    matches: tuple[RequirementMatch, ...],
    *,
    profile_version: str,
    thresholds: Thresholds,
) -> FitReport:
    """Assemble the FitReport. Every derived field is computed here."""
    ordered = _align(matches, role)
    coverage = must_have_coverage(ordered, role)
    score = fit_score(ordered, role)

    return FitReport(
        role_id=role.role_id,
        profile_version=profile_version,
        created_at=datetime.now(UTC),
        matches=ordered,
        must_have_coverage=coverage,
        score=score,
        verdict=verdict_for(coverage, score, thresholds),
        gaps=_gaps(ordered, role),
    )


def _weighted(
    matches: tuple[RequirementMatch, ...], requirements: tuple[Requirement, ...]
) -> float:
    if not requirements:
        return 0.0
    by_id = {m.requirement_id: m for m in matches}
    met = sum(_WEIGHT[by_id[r.id].status] for r in requirements if r.id in by_id)
    return met / len(requirements)


def _align(matches: tuple[RequirementMatch, ...], role: Role) -> tuple[RequirementMatch, ...]:
    """Put matches in role order, inventing an explicit NONE for any the model
    did not answer. A silently missing requirement is the one failure mode that
    would make a report look better than the truth."""
    by_id = {m.requirement_id: m for m in matches}
    return tuple(
        by_id.get(
            r.id,
            RequirementMatch(requirement_id=r.id, status=MatchStatus.NONE, rationale=_UNJUDGED),
        )
        for r in role.requirements
    )


def _gaps(matches: tuple[RequirementMatch, ...], role: Role) -> tuple[str, ...]:
    """Unmet requirements, hard ones first -- that is the order a user cares
    about. Derived from ``matches``; never generated."""
    kinds = {r.id: r.kind for r in role.requirements}
    unmet = [m.requirement_id for m in matches if m.status is MatchStatus.NONE]
    return tuple(sorted(unmet, key=lambda rid: kinds[rid] is not RequirementKind.MUST_HAVE))
