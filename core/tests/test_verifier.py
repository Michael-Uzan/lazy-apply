"""The citation verifier: the grounding guarantee, enforced in code.

A matching model that cites an evidence id which was never sent to it has
invented its justification. The report would still read convincingly, which is
exactly why this is a hard failure rather than a logged warning -- a plausible
ungrounded report is worse than no report.
"""

import pytest

from lazyapply_core.domain import (
    Evidence,
    EvidencePack,
    MatchStatus,
    Requirement,
    RequirementCategory,
    RequirementEvidence,
    RequirementKind,
    ScoredEvidence,
)
from lazyapply_core.evaluate.schema import JudgedRequirement, Judgment
from lazyapply_core.evaluate.verifier import CitationError, verify

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


def requirement(rid: str, category: RequirementCategory = RequirementCategory.SKILL) -> Requirement:
    return Requirement(
        id=rid, text=f"requirement {rid}", kind=RequirementKind.MUST_HAVE, category=category
    )


def pack(*entries: tuple[Requirement, tuple[str, ...]]) -> EvidencePack:
    return EvidencePack(
        role_id="r1",
        items=tuple(
            RequirementEvidence(
                requirement=req,
                hits=tuple(ScoredEvidence(evidence=evidence(s), score=0.5) for s in slugs),
            )
            for req, slugs in entries
        ),
    )


def judged(rid: str, status: MatchStatus, *ids: str) -> JudgedRequirement:
    return JudgedRequirement(
        requirement_id=rid, status=status, evidence_ids=list(ids), rationale="because"
    )


SIMPLE = pack((requirement("a"), ("one", "two")), (requirement("b"), ("two",)))


class TestAcceptance:
    def test_citations_present_in_the_pack_pass(self) -> None:
        verify(Judgment(matches=[judged("a", STRONG, "ev_test__one")]), SIMPLE)

    def test_a_block_retrieved_for_another_requirement_may_be_cited(self) -> None:
        """Blocks are a shared catalogue, not per-requirement allocations. A
        story retrieved for 'b' can genuinely satisfy 'a', and forbidding that
        would discard correct matches."""
        verify(Judgment(matches=[judged("a", STRONG, "ev_test__two")]), SIMPLE)

    def test_none_with_no_citations_passes(self) -> None:
        verify(Judgment(matches=[judged("a", NONE)]), SIMPLE)

    def test_not_every_requirement_has_to_be_judged(self) -> None:
        """A missing judgment is handled by the verdict as an explicit NONE.
        Rejecting the whole response would throw away the judgments that were
        correct."""
        verify(Judgment(matches=[judged("a", STRONG, "ev_test__one")]), SIMPLE)


class TestHallucinatedCitations:
    def test_an_unknown_evidence_id_is_rejected(self) -> None:
        with pytest.raises(CitationError, match="ev_test__invented"):
            verify(Judgment(matches=[judged("a", STRONG, "ev_test__invented")]), SIMPLE)

    def test_the_offending_requirement_is_named(self) -> None:
        with pytest.raises(CitationError, match="a"):
            verify(Judgment(matches=[judged("a", STRONG, "ev_test__invented")]), SIMPLE)

    def test_a_malformed_id_is_rejected_as_a_citation_not_a_schema_error(self) -> None:
        """Whatever shape the model invents, the user sees one clear message."""
        with pytest.raises(CitationError, match="nonsense"):
            verify(Judgment(matches=[judged("a", STRONG, "nonsense")]), SIMPLE)

    def test_one_good_citation_does_not_excuse_a_bad_one(self) -> None:
        with pytest.raises(CitationError, match="ev_test__invented"):
            verify(
                Judgment(matches=[judged("a", STRONG, "ev_test__one", "ev_test__invented")]), SIMPLE
            )


class TestUngroundedClaims:
    def test_strong_without_citations_is_rejected(self) -> None:
        """A positive claim with nothing behind it is the failure this whole
        design exists to prevent."""
        with pytest.raises(CitationError, match="no evidence"):
            verify(Judgment(matches=[judged("a", STRONG)]), SIMPLE)

    def test_partial_without_citations_is_rejected(self) -> None:
        with pytest.raises(CitationError, match="no evidence"):
            verify(Judgment(matches=[judged("a", PARTIAL)]), SIMPLE)

    def test_a_hard_fact_requirement_may_be_met_without_citing_evidence(self) -> None:
        """Education is checked against profile.json, which is injected
        verbatim and carries no evidence ids to cite."""
        facts = pack((requirement("degree", RequirementCategory.EDUCATION), ()))
        verify(Judgment(matches=[judged("degree", STRONG)]), facts)


class TestResponseShape:
    def test_an_unknown_requirement_id_is_rejected(self) -> None:
        """The model answering a requirement that was never asked means it
        drifted from the input; nothing downstream could place the answer."""
        with pytest.raises(CitationError, match="unknown requirement"):
            verify(Judgment(matches=[judged("invented", NONE)]), SIMPLE)

    def test_a_duplicated_requirement_id_is_rejected(self) -> None:
        with pytest.raises(CitationError, match="judged twice"):
            verify(
                Judgment(matches=[judged("a", STRONG, "ev_test__one"), judged("a", NONE)]), SIMPLE
            )
