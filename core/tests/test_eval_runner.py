"""Scoring a pack and a report against the golden labels.

Both functions are pure: they take data the pipeline already produced and the
labels, and return numbers. No embedder, no store, no model -- so the scoring
logic itself is testable without spending anything.
"""

import pytest

from lazyapply_core.config import Thresholds
from lazyapply_core.domain import (
    Evidence,
    EvidencePack,
    MatchStatus,
    Requirement,
    RequirementCategory,
    RequirementEvidence,
    RequirementKind,
    RequirementMatch,
    Role,
    ScoredEvidence,
    Verdict,
)
from lazyapply_core.evals.golden import GoldenCase, GoldenRequirement
from lazyapply_core.evals.runner import (
    LabelDriftError,
    score_matching,
    score_retrieval,
    score_thresholds,
)

THRESHOLDS = Thresholds()

STRONG = MatchStatus.STRONG
PARTIAL = MatchStatus.PARTIAL
NONE = MatchStatus.NONE


def evidence(slug: str) -> Evidence:
    return Evidence(
        evidence_id=f"ev_test__{slug}",
        title=slug,
        body="body",
        composed_text="body",
        content_hash="0" * 16,
        source_file="test.md",
    )


def requirement(rid: str) -> Requirement:
    return Requirement(
        id=rid,
        text=f"did {rid}",
        kind=RequirementKind.MUST_HAVE,
        category=RequirementCategory.SKILL,
    )


def pack(*entries: tuple[str, tuple[str, ...]]) -> EvidencePack:
    return EvidencePack(
        role_id="r1",
        items=tuple(
            RequirementEvidence(
                requirement=requirement(rid),
                hits=tuple(
                    ScoredEvidence(evidence=evidence(s), score=0.5 - i / 100)
                    for i, s in enumerate(slugs)
                ),
            )
            for rid, slugs in entries
        ),
    )


def case(*labels: tuple[str, tuple[str, ...], MatchStatus]) -> GoldenCase:
    return GoldenCase(
        role="test.txt",
        expected_verdict=Verdict.GENERATE,
        requirements=tuple(
            GoldenRequirement(id=rid, relevant=frozenset(rel), status=status)
            for rid, rel, status in labels
        ),
    )


class TestScoreRetrieval:
    def test_a_perfect_retrieval(self) -> None:
        result = score_retrieval(
            case(("a", ("ev_test__one",), STRONG)),
            pack(("a", ("one", "two", "three"))),
            k=3,
        )
        assert result.recall == 1.0
        assert result.mrr == 1.0

    def test_a_miss(self) -> None:
        result = score_retrieval(
            case(("a", ("ev_test__wanted",), STRONG)),
            pack(("a", ("one", "two", "three"))),
            k=3,
        )
        assert result.recall == 0.0
        assert result.mrr == 0.0

    def test_mrr_penalises_a_late_hit_that_recall_forgives(self) -> None:
        result = score_retrieval(
            case(("a", ("ev_test__three",), STRONG)),
            pack(("a", ("one", "two", "three"))),
            k=3,
        )
        assert result.recall == 1.0
        assert result.mrr == pytest.approx(1 / 3)

    def test_averages_across_requirements(self) -> None:
        result = score_retrieval(
            case(("a", ("ev_test__one",), STRONG), ("b", ("ev_test__missing",), STRONG)),
            pack(("a", ("one",)), ("b", ("other",))),
            k=3,
        )
        assert result.recall == 0.5

    def test_unlabelled_requirements_are_excluded_not_failed(self) -> None:
        """Ten firmware requirements have no correct answer. Counting them as
        misses would blame retrieval for questions it was never asked."""
        result = score_retrieval(
            case(("a", ("ev_test__one",), STRONG), ("b", (), NONE)),
            pack(("a", ("one",)), ("b", ("anything",))),
            k=3,
        )
        assert result.recall == 1.0
        assert result.scored == 1
        assert result.unscoreable == 1

    def test_reports_each_requirement_for_drilling_in(self) -> None:
        result = score_retrieval(case(("a", ("ev_test__one",), STRONG)), pack(("a", ("one",))), k=3)
        assert result.details[0].requirement_id == "a"
        assert result.details[0].rank == 1

    def test_rank_is_none_when_nothing_relevant_was_found(self) -> None:
        result = score_retrieval(
            case(("a", ("ev_test__wanted",), STRONG)), pack(("a", ("one",))), k=3
        )
        assert result.details[0].rank is None

    def test_a_set_with_nothing_scoreable_reports_no_number(self) -> None:
        """None rather than 0.0 -- "nothing measured" must not print as a bad
        score."""
        result = score_retrieval(case(("a", (), NONE)), pack(("a", ("one",))), k=3)
        assert result.recall is None
        assert result.mrr is None


class TestScoreMatching:
    def report(self, *statuses: tuple[str, MatchStatus], verdict: Verdict) -> object:
        from datetime import UTC, datetime

        from lazyapply_core.domain import FitReport

        return FitReport(
            role_id="r1",
            profile_version="pv1",
            created_at=datetime.now(UTC),
            matches=tuple(
                RequirementMatch(requirement_id=rid, status=status, rationale="x")
                for rid, status in statuses
            ),
            must_have_coverage=1.0,
            score=100,
            verdict=verdict,
        )

    def test_full_agreement(self) -> None:
        result = score_matching(
            case(("a", (), STRONG), ("b", (), NONE)),
            self.report(("a", STRONG), ("b", NONE), verdict=Verdict.GENERATE),
        )
        assert result.agreement.rate == 1.0

    def test_counts_a_generous_model(self) -> None:
        """The expensive direction: an inflated match is undetectable by
        reading the report."""
        result = score_matching(
            case(("a", (), NONE)),
            self.report(("a", STRONG), verdict=Verdict.GENERATE),
        )
        assert result.agreement.generous == 1
        assert result.agreement.harsh == 0

    def test_compares_the_verdict_too(self) -> None:
        result = score_matching(
            case(("a", (), STRONG)), self.report(("a", STRONG), verdict=Verdict.SKIP)
        )
        assert result.expected_verdict is Verdict.GENERATE
        assert result.actual_verdict is Verdict.SKIP
        assert not result.verdict_matched


class TestLabelDrift:
    def test_a_label_for_a_requirement_that_no_longer_exists_is_an_error(self) -> None:
        """Re-extraction can rename requirements. Silently skipping the stale
        label would quietly shrink the measured set and improve the numbers."""
        with pytest.raises(LabelDriftError, match="renamed"):
            score_retrieval(
                case(("renamed", ("ev_test__one",), STRONG)), pack(("a", ("one",))), k=3
            )

    def test_an_unlabelled_requirement_in_the_role_is_an_error(self) -> None:
        with pytest.raises(LabelDriftError, match="unlabelled"):
            score_retrieval(
                case(("a", ("ev_test__one",), STRONG)), pack(("a", ("one",)), ("b", ())), k=3
            )


class TestThresholdMargins:
    """What the thresholds are actually being asked to decide.

    Tuning three numbers needs postings that land near them. This measures
    where each posting falls, from the *human* labels, so an under-powered
    golden set is visible rather than asserted.
    """

    def role(self, *specs: tuple[str, RequirementKind]) -> Role:
        return Role(
            role_id="r1",
            title="Test",
            raw_text="...",
            requirements=tuple(
                Requirement(
                    id=rid, text=f"did {rid}", kind=kind, category=RequirementCategory.SKILL
                )
                for rid, kind in specs
            ),
        )

    def test_computes_coverage_from_the_human_labels(self) -> None:
        margin = score_thresholds(
            case(("a", (), STRONG), ("b", (), NONE)),
            self.role(("a", RequirementKind.MUST_HAVE), ("b", RequirementKind.MUST_HAVE)),
            THRESHOLDS,
        )
        assert margin.coverage == 0.5

    def test_derives_the_verdict_a_person_would_reach(self) -> None:
        margin = score_thresholds(
            case(("a", (), STRONG)), self.role(("a", RequirementKind.MUST_HAVE)), THRESHOLDS
        )
        assert margin.verdict is Verdict.GENERATE

    def test_reports_the_distance_to_the_nearest_boundary(self) -> None:
        """A posting at 1.0 coverage says nothing about whether 0.80 is the
        right cutoff. The distance is how much it says."""
        margin = score_thresholds(
            case(("a", (), STRONG)), self.role(("a", RequirementKind.MUST_HAVE)), THRESHOLDS
        )
        assert margin.distance_to_boundary == pytest.approx(0.20)

    def test_a_posting_near_a_boundary_reports_a_small_distance(self) -> None:
        margin = score_thresholds(
            case(("a", (), STRONG), ("b", (), PARTIAL), ("c", (), NONE), ("d", (), STRONG)),
            self.role(*((rid, RequirementKind.MUST_HAVE) for rid in "abcd")),
            THRESHOLDS,
        )
        assert margin.coverage == pytest.approx(0.625)
        assert margin.distance_to_boundary == pytest.approx(0.025)


class TestScoringAtSeveralCutoffs:
    """One pack, several values of k.

    A retrieved vector does not depend on k -- only the cutoff does. Scoring a
    single pack at k=1..5 answers "what should top_k be" from one embedding
    request instead of five, which matters on a 3 RPM free tier and matters
    more for not paying five times for the same answer.
    """

    def test_a_tighter_cutoff_can_lose_a_hit(self) -> None:
        labels = case(("a", ("ev_test__three",), STRONG))
        retrieved = pack(("a", ("one", "two", "three")))

        assert score_retrieval(labels, retrieved, k=3).recall == 1.0
        assert score_retrieval(labels, retrieved, k=2).recall == 0.0

    def test_a_wider_cutoff_can_recover_one(self) -> None:
        labels = case(("a", ("ev_test__four",), STRONG))
        retrieved = pack(("a", ("one", "two", "three", "four")))

        assert score_retrieval(labels, retrieved, k=3).recall == 0.0
        assert score_retrieval(labels, retrieved, k=4).recall == 1.0

    def test_mrr_is_unaffected_by_a_hit_beyond_the_cutoff(self) -> None:
        """The reader never sees past k, so a hit below it cannot shorten the
        search."""
        labels = case(("a", ("ev_test__one", "ev_test__four"), STRONG))
        retrieved = pack(("a", ("one", "two", "three", "four")))

        assert score_retrieval(labels, retrieved, k=3).mrr == 1.0
        assert score_retrieval(labels, retrieved, k=4).mrr == 1.0
