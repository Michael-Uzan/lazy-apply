"""Rendering the matching prompt: hard facts, evidence catalogue, requirements.

The shape matters more than the wording. Every block appears once and is
referred to by id, so a block retrieved for five requirements does not consume
the context five times -- and so the model has an id to cite that the verifier
can check.
"""

import pytest

from lazyapply_core.domain import (
    Education,
    Evidence,
    EvidencePack,
    Position,
    Profile,
    Requirement,
    RequirementCategory,
    RequirementEvidence,
    RequirementKind,
    ScoredEvidence,
)
from lazyapply_core.evaluate.render import render_facts, render_pack


@pytest.fixture
def profile() -> Profile:
    return Profile(
        name="Dana Okonkwo",
        headline="Senior Full-Stack Engineer",
        location="Lisbon, Portugal",
        email="dana@example.com",
        education=(
            Education(
                degree="B.Sc. Computer Science",
                institution="University of Lisbon",
                graduated="2016-07",
            ),
        ),
        positions=(
            Position(
                company="Northwind Trading",
                title="Senior Full-Stack Engineer",
                start="2022-03",
                end="present",
            ),
            Position(company="Kestrel", title="Engineer", start="2019-01", end="2022-02"),
        ),
        stack=("react", "typescript"),
    )


def evidence(slug: str, body: str = "body text") -> Evidence:
    return Evidence(
        evidence_id=f"ev_test__{slug}",
        title=slug.replace("-", " "),
        body=body,
        composed_text=f"Test — {slug}\n{body}",
        content_hash="0" * 16,
        company="Northwind Trading",
        tags=("react",),
        source_file="test.md",
    )


def requirement(
    rid: str,
    kind: RequirementKind = RequirementKind.MUST_HAVE,
    category: RequirementCategory = RequirementCategory.SKILL,
) -> Requirement:
    return Requirement(id=rid, text=f"did the thing called {rid}", kind=kind, category=category)


def pack(*entries: tuple[Requirement, tuple[tuple[str, float], ...]]) -> EvidencePack:
    return EvidencePack(
        role_id="r1",
        items=tuple(
            RequirementEvidence(
                requirement=req,
                hits=tuple(ScoredEvidence(evidence=evidence(s), score=sc) for s, sc in hits),
            )
            for req, hits in entries
        ),
    )


class TestFacts:
    def test_includes_the_name_and_headline(self, profile: Profile) -> None:
        text = render_facts(profile)
        assert "Dana Okonkwo" in text
        assert "Senior Full-Stack Engineer" in text

    def test_includes_the_degree_and_institution(self, profile: Profile) -> None:
        """This is the whole point of hard facts: a degree requirement is
        answered here, verbatim, and never by a similarity search."""
        text = render_facts(profile)
        assert "B.Sc. Computer Science" in text
        assert "University of Lisbon" in text

    def test_includes_employment_dates(self, profile: Profile) -> None:
        text = render_facts(profile)
        assert "2022-03" in text
        assert "present" in text
        assert "2019-01" in text

    def test_includes_the_stack(self, profile: Profile) -> None:
        assert "typescript" in render_facts(profile)

    def test_omits_empty_sections(self) -> None:
        """A profile with no education should not carry an empty heading the
        model might try to fill in."""
        bare = Profile(name="A", headline="B", location="C", email="a@example.com")
        assert "Education" not in render_facts(bare)


class TestEvidenceCatalogue:
    def test_each_block_body_appears_exactly_once(self) -> None:
        """The same block is usually retrieved for several requirements. The
        catalogue carries each body once; requirements refer to ids."""
        shared = pack(
            (requirement("a"), (("one", 0.5), ("two", 0.4))),
            (requirement("b"), (("one", 0.6),)),
        )
        text = render_pack(shared)
        assert text.count("body text") == 2

    def test_every_retrieved_block_is_in_the_catalogue(self) -> None:
        text = render_pack(pack((requirement("a"), (("one", 0.5), ("two", 0.4)))))
        catalogue = text.split("<requirements>")[0]
        assert "ev_test__one" in catalogue
        assert "ev_test__two" in catalogue

    def test_the_catalogue_is_empty_when_nothing_was_retrieved(self) -> None:
        """A role of pure hard facts retrieves nothing, and the prompt must
        still be coherent rather than carrying a dangling heading."""
        text = render_pack(pack((requirement("a"), ())))
        assert "ev_test__" not in text


class TestRequirements:
    def test_lists_every_requirement_with_its_id(self) -> None:
        text = render_pack(pack((requirement("a"), ()), (requirement("b"), ())))
        assert "a" in text and "b" in text

    def test_marks_must_have_and_nice_to_have(self) -> None:
        text = render_pack(
            pack(
                (requirement("hard", RequirementKind.MUST_HAVE), ()),
                (requirement("soft", RequirementKind.NICE_TO_HAVE), ()),
            )
        )
        assert "must-have" in text
        assert "nice-to-have" in text

    def test_names_the_blocks_retrieved_for_each_requirement(self) -> None:
        """Candidates are listed per requirement, so the model knows where to
        look first -- but it may cite anything in the catalogue."""
        text = render_pack(pack((requirement("a"), (("one", 0.5),)), (requirement("b"), ())))
        lines = text.splitlines()
        a_line = next(i for i, line in enumerate(lines) if "did the thing called a" in line)
        assert "ev_test__one" in lines[a_line + 1]

    def test_says_when_a_requirement_retrieved_nothing(self) -> None:
        """An empty requirement is the signal for a gap. It must be visible as
        absence, not as a blank line the model might overlook."""
        text = render_pack(pack((requirement("a"), ())))
        assert "nothing retrieved" in text

    def test_hard_facts_are_marked_and_point_at_the_facts_block(self) -> None:
        degree = requirement("degree", category=RequirementCategory.EDUCATION)
        text = render_pack(pack((degree, ())))
        assert "nothing retrieved" not in text
        assert "facts" in text.lower()
