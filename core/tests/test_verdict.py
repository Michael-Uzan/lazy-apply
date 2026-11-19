"""The verdict: pure aggregation over the model's per-requirement statuses.

The whole point of this module is that it contains no model call and no I/O.
A verdict can therefore be reproduced exactly, swept against the golden set,
and explained to a user -- "you are missing 3 of 7 must-haves" -- which is
impossible if a model emitted the decision itself.
"""

import pytest

from lazyapply_core.config import Thresholds
from lazyapply_core.domain import (
    MatchStatus,
    Requirement,
    RequirementCategory,
    RequirementKind,
    RequirementMatch,
    Role,
    Verdict,
)
from lazyapply_core.evaluate.verdict import (
    build_report,
    fit_score,
    must_have_coverage,
    verdict_for,
)

THRESHOLDS = Thresholds()


def role(*specs: tuple[str, RequirementKind]) -> Role:
    return Role(
        role_id="r1",
        title="Test Role",
        raw_text="...",
        requirements=tuple(
            Requirement(
                id=rid,
                text=f"requirement {rid}",
                kind=kind,
                category=RequirementCategory.SKILL,
            )
            for rid, kind in specs
        ),
    )


def musts(*ids: str) -> Role:
    return role(*((rid, RequirementKind.MUST_HAVE) for rid in ids))


def match(rid: str, status: MatchStatus) -> RequirementMatch:
    return RequirementMatch(requirement_id=rid, status=status, rationale="because")


def matches(**statuses: MatchStatus) -> tuple[RequirementMatch, ...]:
    return tuple(match(rid, status) for rid, status in statuses.items())


STRONG = MatchStatus.STRONG
PARTIAL = MatchStatus.PARTIAL
NONE = MatchStatus.NONE


class TestMustHaveCoverage:
    def test_all_strong_is_full_coverage(self) -> None:
        assert must_have_coverage(matches(a=STRONG, b=STRONG), musts("a", "b")) == 1.0

    def test_all_none_is_zero_coverage(self) -> None:
        assert must_have_coverage(matches(a=NONE, b=NONE), musts("a", "b")) == 0.0

    def test_a_partial_counts_as_half(self) -> None:
        """Partial evidence is real but weaker. Counting it as zero would
        reject candidates who can argue the point; counting it as one would
        send them into interviews unprepared."""
        assert must_have_coverage(matches(a=PARTIAL, b=PARTIAL), musts("a", "b")) == 0.5

    def test_mixed_statuses(self) -> None:
        got = must_have_coverage(matches(a=STRONG, b=PARTIAL, c=NONE, d=STRONG), musts(*"abcd"))
        assert got == pytest.approx(0.625)

    def test_nice_to_haves_do_not_affect_coverage(self) -> None:
        """Coverage answers one question: am I allowed to apply? Only hard
        requirements can block that."""
        r = role(("a", RequirementKind.MUST_HAVE), ("b", RequirementKind.NICE_TO_HAVE))
        assert must_have_coverage(matches(a=STRONG, b=NONE), r) == 1.0

    def test_an_unjudged_must_have_counts_as_missing(self) -> None:
        """A requirement the model skipped is not silently dropped from the
        denominator -- that would inflate coverage by omission."""
        assert must_have_coverage(matches(a=STRONG), musts("a", "b")) == 0.5


class TestFitScore:
    def test_is_coverage_when_there_are_no_nice_to_haves(self) -> None:
        assert fit_score(matches(a=STRONG, b=NONE), musts("a", "b")) == 50

    def test_must_haves_dominate(self) -> None:
        """Every nice-to-have met cannot rescue a candidate who meets no hard
        requirement."""
        r = role(("a", RequirementKind.MUST_HAVE), ("b", RequirementKind.NICE_TO_HAVE))
        assert fit_score(matches(a=NONE, b=STRONG), r) == 20

    def test_nice_to_haves_separate_two_candidates_who_both_clear_the_bar(self) -> None:
        r = role(("a", RequirementKind.MUST_HAVE), ("b", RequirementKind.NICE_TO_HAVE))
        assert fit_score(matches(a=STRONG, b=NONE), r) == 80
        assert fit_score(matches(a=STRONG, b=STRONG), r) == 100

    def test_is_an_integer_in_range(self) -> None:
        r = role(("a", RequirementKind.MUST_HAVE), ("b", RequirementKind.NICE_TO_HAVE))
        got = fit_score(matches(a=PARTIAL, b=PARTIAL), r)
        assert isinstance(got, int)
        assert 0 <= got <= 100


class TestVerdictFor:
    def test_below_skip_threshold_is_skip(self) -> None:
        assert verdict_for(0.5, 100, THRESHOLDS) is Verdict.SKIP

    def test_full_coverage_and_a_good_score_generates(self) -> None:
        assert verdict_for(1.0, 100, THRESHOLDS) is Verdict.GENERATE

    def test_between_the_thresholds_is_borderline(self) -> None:
        assert verdict_for(0.7, 100, THRESHOLDS) is Verdict.BORDERLINE

    def test_good_coverage_but_a_weak_score_is_borderline(self) -> None:
        """Coverage can be met entirely by PARTIALs; the score catches that."""
        assert verdict_for(0.9, 50, THRESHOLDS) is Verdict.BORDERLINE

    def test_the_thresholds_are_inclusive_at_their_boundaries(self) -> None:
        assert verdict_for(0.60, 100, THRESHOLDS) is Verdict.BORDERLINE
        assert verdict_for(0.80, 70, THRESHOLDS) is Verdict.GENERATE

    def test_thresholds_are_configurable(self) -> None:
        strict = Thresholds(skip_below=0.95, generate_at=1.0, min_score=90)
        assert verdict_for(0.9, 100, strict) is Verdict.SKIP
        assert verdict_for(1.0, 100, strict) is Verdict.GENERATE


class TestBuildReport:
    def test_carries_the_role_and_profile_identity(self) -> None:
        report = build_report(
            musts("a"), matches(a=STRONG), profile_version="pv1", thresholds=THRESHOLDS
        )
        assert report.role_id == "r1"
        assert report.profile_version == "pv1"

    def test_computes_coverage_score_and_verdict(self) -> None:
        report = build_report(
            musts("a", "b"),
            matches(a=STRONG, b=STRONG),
            profile_version="pv1",
            thresholds=THRESHOLDS,
        )
        assert report.must_have_coverage == 1.0
        assert report.score == 100
        assert report.verdict is Verdict.GENERATE

    def test_gaps_are_derived_from_the_matches(self) -> None:
        """Never generated by the model -- a gap list that disagreed with the
        statuses it came from would be unexplainable."""
        report = build_report(
            musts("a", "b", "c"),
            matches(a=STRONG, b=NONE, c=NONE),
            profile_version="pv1",
            thresholds=THRESHOLDS,
        )
        assert report.gaps == ("b", "c")

    def test_must_have_gaps_come_before_nice_to_have_gaps(self) -> None:
        r = role(("nice", RequirementKind.NICE_TO_HAVE), ("must", RequirementKind.MUST_HAVE))
        report = build_report(
            r, matches(nice=NONE, must=NONE), profile_version="pv1", thresholds=THRESHOLDS
        )
        assert report.gaps == ("must", "nice")

    def test_partial_is_not_a_gap(self) -> None:
        report = build_report(
            musts("a"), matches(a=PARTIAL), profile_version="pv1", thresholds=THRESHOLDS
        )
        assert report.gaps == ()

    def test_matches_keep_the_role_order(self) -> None:
        report = build_report(
            musts("a", "b", "c"),
            matches(c=STRONG, a=STRONG, b=STRONG),
            profile_version="pv1",
            thresholds=THRESHOLDS,
        )
        assert [m.requirement_id for m in report.matches] == ["a", "b", "c"]

    def test_an_unjudged_requirement_becomes_an_explicit_none(self) -> None:
        """Silence from the model is recorded as unmet, with a rationale that
        says so, rather than vanishing from the report."""
        report = build_report(
            musts("a", "b"), matches(a=STRONG), profile_version="pv1", thresholds=THRESHOLDS
        )
        unjudged = next(m for m in report.matches if m.requirement_id == "b")
        assert unjudged.status is MatchStatus.NONE
        assert "not judged" in unjudged.rationale.lower()

    def test_created_at_is_recorded(self) -> None:
        report = build_report(
            musts("a"), matches(a=STRONG), profile_version="pv1", thresholds=THRESHOLDS
        )
        assert report.created_at.tzinfo is not None
