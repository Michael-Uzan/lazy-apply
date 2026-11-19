"""Loading and validating the hand-labelled golden set.

The labels are the only thing in this project that cannot be regenerated, so
the loader is strict: a typo in an evidence id would otherwise show up as a
retrieval failure and send someone hunting a bug in the embedder.
"""

from pathlib import Path

import pytest

from lazyapply_core.domain import MatchStatus, Verdict
from lazyapply_core.evals.golden import GoldenError, load_golden

VALID = """
version: 1
cases:
  - role: strong-match.txt
    expected_verdict: generate
    requirements:
      - id: react_production
        relevant: [ev_northwind__real-time-trading-dashboard]
        status: strong
      - id: embedded_c
        relevant: []
        status: none
        note: nothing in the evidence touches firmware
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "golden.yaml"
    path.write_text(text)
    return path


class TestLoading:
    def test_reads_every_case(self, tmp_path: Path) -> None:
        golden = load_golden(write(tmp_path, VALID))
        assert len(golden.cases) == 1

    def test_reads_every_requirement(self, tmp_path: Path) -> None:
        assert len(load_golden(write(tmp_path, VALID)).cases[0].requirements) == 2

    def test_parses_the_relevant_set(self, tmp_path: Path) -> None:
        requirement = load_golden(write(tmp_path, VALID)).cases[0].requirements[0]
        assert requirement.relevant == {"ev_northwind__real-time-trading-dashboard"}

    def test_parses_the_expected_status(self, tmp_path: Path) -> None:
        requirement = load_golden(write(tmp_path, VALID)).cases[0].requirements[0]
        assert requirement.status is MatchStatus.STRONG

    def test_parses_the_expected_verdict(self, tmp_path: Path) -> None:
        assert load_golden(write(tmp_path, VALID)).cases[0].expected_verdict is Verdict.GENERATE

    def test_an_empty_relevant_set_is_allowed(self, tmp_path: Path) -> None:
        """A requirement with no correct answer is a label, not an omission --
        it is how the set tests that the system declines to match."""
        assert load_golden(write(tmp_path, VALID)).cases[0].requirements[1].relevant == set()

    def test_notes_are_optional_and_kept(self, tmp_path: Path) -> None:
        requirements = load_golden(write(tmp_path, VALID)).cases[0].requirements
        assert requirements[0].note is None
        assert "firmware" in str(requirements[1].note)


class TestScoreable:
    def test_separates_requirements_that_can_be_scored_for_recall(self, tmp_path: Path) -> None:
        """Dividing by an empty relevant set is undefined, so these are
        reported separately rather than silently counted as failures."""
        case = load_golden(write(tmp_path, VALID)).cases[0]
        assert [r.id for r in case.scoreable] == ["react_production"]


class TestErrors:
    def test_a_missing_file_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(GoldenError, match="not found"):
            load_golden(tmp_path / "nope.yaml")

    def test_invalid_yaml_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(GoldenError, match="not valid YAML"):
            load_golden(write(tmp_path, "cases: [oh: no: yes"))

    def test_an_unknown_version_is_refused(self, tmp_path: Path) -> None:
        """Changing the label format silently would invalidate every number
        already published from it."""
        with pytest.raises(GoldenError, match="version"):
            load_golden(write(tmp_path, "version: 99\ncases: []"))

    def test_an_unknown_status_is_reported_by_name(self, tmp_path: Path) -> None:
        bad = VALID.replace("status: strong", "status: excellent", 1)
        with pytest.raises(GoldenError, match="excellent"):
            load_golden(write(tmp_path, bad))

    def test_a_malformed_evidence_id_is_reported(self, tmp_path: Path) -> None:
        bad = VALID.replace("ev_northwind__real-time-trading-dashboard", "northwind-dashboard")
        with pytest.raises(GoldenError, match="northwind-dashboard"):
            load_golden(write(tmp_path, bad))

    def test_a_duplicate_requirement_id_is_reported(self, tmp_path: Path) -> None:
        bad = VALID.replace("id: embedded_c", "id: react_production")
        with pytest.raises(GoldenError, match="react_production"):
            load_golden(write(tmp_path, bad))

    def test_a_case_with_no_requirements_is_reported(self, tmp_path: Path) -> None:
        bad = (
            "version: 1\ncases:\n  - role: a.txt\n    expected_verdict: skip\n    requirements: []"
        )
        with pytest.raises(GoldenError, match="no requirements"):
            load_golden(write(tmp_path, bad))


class TestTheRealGoldenSet:
    def test_the_committed_set_loads(self) -> None:
        """It is hand-edited, so it is worth a test that it still parses."""
        from lazyapply_core.evals.golden import DEFAULT_GOLDEN_PATH

        golden = load_golden(DEFAULT_GOLDEN_PATH)
        assert len(golden.cases) == 2

    def test_it_labels_every_requirement_of_both_postings(self) -> None:
        from lazyapply_core.evals.golden import DEFAULT_GOLDEN_PATH

        golden = load_golden(DEFAULT_GOLDEN_PATH)
        assert sum(len(c.requirements) for c in golden.cases) == 21
