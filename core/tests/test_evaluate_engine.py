"""The fit pipeline end to end, with the model faked.

Three steps in order: the model judges, the verifier checks the grounding,
pure code computes the verdict. The ordering is the design -- an ungrounded
judgment must never reach the verdict, because a verdict is what the user
acts on.
"""

import pytest

from lazyapply_core.config import Thresholds
from lazyapply_core.domain import (
    Evidence,
    EvidencePack,
    MatchStatus,
    Profile,
    Requirement,
    RequirementCategory,
    RequirementEvidence,
    RequirementKind,
    Role,
    ScoredEvidence,
    Verdict,
)
from lazyapply_core.evaluate.base import EvaluationError
from lazyapply_core.evaluate.engine import evaluate_fit
from lazyapply_core.evaluate.fake import FakeEvaluator
from lazyapply_core.evaluate.schema import JudgedRequirement, Judgment
from lazyapply_core.evaluate.verifier import CitationError

THRESHOLDS = Thresholds()

PROFILE = Profile(
    name="Dana Okonkwo", headline="Engineer", location="Lisbon", email="dana@example.com"
)


def evidence(slug: str) -> Evidence:
    return Evidence(
        evidence_id=f"ev_test__{slug}",
        title=slug,
        body="body",
        composed_text="body",
        content_hash="0" * 16,
        source_file="test.md",
    )


def requirement(rid: str, kind: RequirementKind = RequirementKind.MUST_HAVE) -> Requirement:
    return Requirement(id=rid, text=f"did {rid}", kind=kind, category=RequirementCategory.SKILL)


ROLE = Role(
    role_id="r1",
    title="Senior Engineer",
    raw_text="...",
    requirements=(requirement("a"), requirement("b")),
)

PACK = EvidencePack(
    role_id="r1",
    items=(
        RequirementEvidence(
            requirement=requirement("a"),
            hits=(ScoredEvidence(evidence=evidence("one"), score=0.56),),
        ),
        RequirementEvidence(requirement=requirement("b"), hits=()),
    ),
)


def judged(rid: str, status: MatchStatus, *ids: str) -> JudgedRequirement:
    return JudgedRequirement(
        requirement_id=rid, status=status, evidence_ids=list(ids), rationale=f"about {rid}"
    )


def run(*judgments: JudgedRequirement, thresholds: Thresholds = THRESHOLDS):  # type: ignore[no-untyped-def]
    return evaluate_fit(
        ROLE,
        PACK,
        PROFILE,
        evaluator=FakeEvaluator(Judgment(matches=list(judgments))),
        thresholds=thresholds,
        profile_version="pv1",
    )


class TestHappyPath:
    def test_produces_a_report_for_the_role(self) -> None:
        report = run(judged("a", MatchStatus.STRONG, "ev_test__one"), judged("b", MatchStatus.NONE))
        assert report.role_id == "r1"
        assert report.profile_version == "pv1"

    def test_the_verdict_is_computed_not_asked_for(self) -> None:
        """Nothing in the model's response names a verdict. Half the
        must-haves met is below the skip threshold."""
        report = run(judged("a", MatchStatus.STRONG, "ev_test__one"), judged("b", MatchStatus.NONE))
        assert report.must_have_coverage == 0.5
        assert report.verdict is Verdict.SKIP

    def test_a_fully_met_role_generates(self) -> None:
        report = run(
            judged("a", MatchStatus.STRONG, "ev_test__one"),
            judged("b", MatchStatus.STRONG, "ev_test__one"),
        )
        assert report.verdict is Verdict.GENERATE

    def test_citations_survive_into_the_report(self) -> None:
        """The report has to be auditable: a reader can look up every id."""
        report = run(judged("a", MatchStatus.STRONG, "ev_test__one"), judged("b", MatchStatus.NONE))
        assert report.matches[0].evidence_ids == ("ev_test__one",)

    def test_rationales_survive_into_the_report(self) -> None:
        report = run(judged("a", MatchStatus.STRONG, "ev_test__one"), judged("b", MatchStatus.NONE))
        assert report.matches[0].rationale == "about a"

    def test_the_retrieval_score_is_attached_from_the_pack(self) -> None:
        """Diagnostic for threshold tuning. It comes from retrieval, never
        from the model -- a model has no way to know a cosine distance."""
        report = run(judged("a", MatchStatus.STRONG, "ev_test__one"), judged("b", MatchStatus.NONE))
        assert report.matches[0].best_retrieval_score == pytest.approx(0.56)
        assert report.matches[1].best_retrieval_score is None

    def test_gaps_name_the_unmet_requirements(self) -> None:
        report = run(judged("a", MatchStatus.STRONG, "ev_test__one"), judged("b", MatchStatus.NONE))
        assert report.gaps == ("b",)

    def test_a_requirement_the_model_skipped_becomes_a_gap(self) -> None:
        report = run(judged("a", MatchStatus.STRONG, "ev_test__one"))
        assert report.gaps == ("b",)


class TestFailuresStopBeforeTheVerdict:
    def test_a_hallucinated_citation_raises(self) -> None:
        """An ungrounded judgment must never reach a verdict. The report would
        read convincingly, which is exactly the danger."""
        with pytest.raises(CitationError, match="ev_test__invented"):
            run(judged("a", MatchStatus.STRONG, "ev_test__invented"))

    def test_an_ungrounded_strong_raises(self) -> None:
        with pytest.raises(CitationError, match="no evidence"):
            run(judged("a", MatchStatus.STRONG))

    def test_a_provider_failure_propagates(self) -> None:
        failing = FakeEvaluator(Judgment(matches=[]), error=EvaluationError("rate limited"))
        with pytest.raises(EvaluationError, match="rate limited"):
            evaluate_fit(
                ROLE,
                PACK,
                PROFILE,
                evaluator=failing,
                thresholds=THRESHOLDS,
                profile_version="pv1",
            )


class TestOneCall:
    def test_the_whole_role_is_judged_in_a_single_request(self) -> None:
        """Per-requirement calls would multiply the cost of every check by ten
        for no gain -- the model sees all the evidence either way."""
        evaluator = FakeEvaluator(
            Judgment(matches=[judged("a", MatchStatus.STRONG, "ev_test__one")])
        )
        evaluate_fit(
            ROLE,
            PACK,
            PROFILE,
            evaluator=evaluator,
            thresholds=THRESHOLDS,
            profile_version="pv1",
        )
        assert evaluator.calls == 1
