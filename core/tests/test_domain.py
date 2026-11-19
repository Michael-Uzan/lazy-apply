import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from lazyapply_core.config import REPO_ROOT
from lazyapply_core.domain import (
    Evidence,
    EvidenceId,
    Profile,
    Requirement,
    RequirementCategory,
    RequirementKind,
    Role,
    ScoredEvidence,
    make_evidence_id,
)


class TestEvidence:
    def test_is_immutable(self, evidence: Evidence) -> None:
        with pytest.raises(ValidationError):
            evidence.title = "something else"  # type: ignore[misc]

    def test_is_hashable(self, evidence: Evidence) -> None:
        assert len({evidence, evidence}) == 1

    @pytest.mark.parametrize(
        "bad_id",
        [
            "northwind__dashboard",  # missing ev_ prefix
            "ev_northwind_dashboard",  # single underscore separator
            "ev_Northwind__Dashboard",  # uppercase
            "ev_northwind__",  # empty slug
        ],
    )
    def test_rejects_malformed_evidence_id(self, evidence: Evidence, bad_id: str) -> None:
        """A typo'd citation anchor must fail at construction, not silently."""
        with pytest.raises(ValidationError):
            Evidence.model_validate(evidence.model_dump() | {"evidence_id": bad_id})

    @pytest.mark.parametrize("bad_date", ["2022-3", "2022-13", "2022", "22-03"])
    def test_rejects_malformed_dates(self, evidence: Evidence, bad_date: str) -> None:
        with pytest.raises(ValidationError):
            Evidence.model_validate(evidence.model_dump() | {"start": bad_date})

    def test_accepts_present_as_end_date(self, evidence: Evidence) -> None:
        assert Evidence.model_validate(evidence.model_dump() | {"end": "present"}).end == "present"


class TestRole:
    def _role(self) -> Role:
        return Role(
            role_id="abc123",
            title="Senior Full-Stack Engineer",
            raw_text="...",
            requirements=(
                Requirement(
                    id="req_react",
                    text="Built production web applications with React",
                    kind=RequirementKind.MUST_HAVE,
                    category=RequirementCategory.SKILL,
                ),
                Requirement(
                    id="req_fintech",
                    text="Worked in fintech or trading systems",
                    kind=RequirementKind.NICE_TO_HAVE,
                    category=RequirementCategory.DOMAIN,
                ),
            ),
        )

    def test_splits_must_haves_from_nice_to_haves(self) -> None:
        role = self._role()
        assert [r.id for r in role.must_haves] == ["req_react"]
        assert [r.id for r in role.nice_to_haves] == ["req_fintech"]


class TestProfile:
    def test_example_profile_parses(self) -> None:
        """The committed sample must stay valid — it is the quickstart."""
        raw = json.loads(Path(REPO_ROOT / "data/example/profile.json").read_text())
        profile = Profile.model_validate(raw)
        assert profile.name == "Dana Levi"
        assert profile.positions[0].end == "present"


class TestScoredEvidence:
    """Cosine similarity is mathematically in [-1, 1], but floating-point
    arithmetic overshoots: a unit vector dotted with itself yields
    1.0000000000000002. Tiny overshoot is noise and is clamped; a genuinely
    out-of-range score is a bug and still fails."""

    def _scored(self, evidence: Evidence, score: float) -> ScoredEvidence:
        return ScoredEvidence(evidence=evidence, score=score)

    def test_clamps_floating_point_overshoot_above_one(self, evidence: Evidence) -> None:
        assert self._scored(evidence, 1.0000000000000002).score == 1.0

    def test_clamps_floating_point_overshoot_below_minus_one(self, evidence: Evidence) -> None:
        assert self._scored(evidence, -1.0000000000000002).score == -1.0

    def test_leaves_ordinary_scores_alone(self, evidence: Evidence) -> None:
        assert self._scored(evidence, 0.5843).score == 0.5843

    @pytest.mark.parametrize("bad_score", [1.5, -2.0, 100.0])
    def test_rejects_scores_that_are_actually_wrong(
        self, evidence: Evidence, bad_score: float
    ) -> None:
        with pytest.raises(ValidationError):
            self._scored(evidence, bad_score)


class TestRequirementKindLabel:
    """One display label, defined once.

    Four modules were formatting this independently -- two identical
    ternaries, a ``.replace("_", "-")``, and a literal pair -- and one of them
    then compared a formatted label back against a string literal to count
    unmet must-haves. That comparison would have broken silently if any of the
    four had been reworded.
    """

    def test_must_have_reads_as_prose(self) -> None:
        assert RequirementKind.MUST_HAVE.label == "must-have"

    def test_nice_to_have_reads_as_prose(self) -> None:
        assert RequirementKind.NICE_TO_HAVE.label == "nice-to-have"

    def test_every_kind_has_one(self) -> None:
        """A new kind must not silently render as a raw enum value."""
        assert all("_" not in kind.label for kind in RequirementKind)


class TestMakeEvidenceId:
    """The id format has one definition, shared with the pattern that validates it."""

    def test_joins_the_file_and_title_slugs(self) -> None:
        assert make_evidence_id("leverate", "multi-agent-orchestration") == (
            "ev_leverate__multi-agent-orchestration"
        )

    def test_the_result_satisfies_the_evidence_id_pattern(self) -> None:
        """Format and validation drifting apart is the failure this guards."""
        built = make_evidence_id("northwind", "trading-dashboard")
        assert TypeAdapter(EvidenceId).validate_python(built) == built
