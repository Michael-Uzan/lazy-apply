import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lazyapply_core.config import REPO_ROOT
from lazyapply_core.domain import (
    Evidence,
    Profile,
    Requirement,
    RequirementCategory,
    RequirementKind,
    Role,
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
