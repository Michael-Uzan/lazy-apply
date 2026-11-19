"""Loading profile.json — the hard facts.

These never pass through retrieval and are never paraphrased by a model, so
the only thing that can corrupt them is a bad file. Validation is therefore
strict and the errors name the field.
"""

import json
from pathlib import Path

import pytest

from lazyapply_core.config import REPO_ROOT
from lazyapply_core.ingest.parser import ParseError
from lazyapply_core.ingest.profile import load_profile, profile_version

EXAMPLE = REPO_ROOT / "data/example/profile.json"


def write(path: Path, **overrides: object) -> Path:
    data = json.loads(EXAMPLE.read_text()) | overrides
    target = path / "profile.json"
    target.write_text(json.dumps(data))
    return target


class TestLoading:
    def test_loads_the_committed_example(self) -> None:
        assert load_profile(EXAMPLE).name == "Dana Levi"

    def test_reads_nested_structures(self) -> None:
        profile = load_profile(EXAMPLE)
        assert profile.education[0].institution == "Technion"
        assert profile.positions[0].end == "present"
        assert "TypeScript" in profile.stack

    def test_links_are_parsed_as_urls(self) -> None:
        labels = {link.label for link in load_profile(EXAMPLE).links}
        assert labels == {"GitHub", "LinkedIn"}


class TestFailsLoudly:
    def test_missing_file_names_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(ParseError, match=r"profile\.json"):
            load_profile(tmp_path / "profile.json")

    def test_invalid_json_is_reported_as_such(self, tmp_path: Path) -> None:
        bad = tmp_path / "profile.json"
        bad.write_text("{not json")
        with pytest.raises(ParseError, match="not valid JSON"):
            load_profile(bad)

    def test_rejects_a_malformed_email(self, tmp_path: Path) -> None:
        with pytest.raises(ParseError, match="email"):
            load_profile(write(tmp_path, email="not-an-email"))

    def test_rejects_a_malformed_date(self, tmp_path: Path) -> None:
        with pytest.raises(ParseError, match="graduated"):
            load_profile(
                write(
                    tmp_path,
                    education=[{"degree": "B.Sc.", "institution": "X", "graduated": "2018-13"}],
                )
            )

    def test_rejects_a_missing_required_field(self, tmp_path: Path) -> None:
        data = json.loads(EXAMPLE.read_text())
        del data["name"]
        target = tmp_path / "profile.json"
        target.write_text(json.dumps(data))
        with pytest.raises(ParseError, match="name"):
            load_profile(target)


class TestProfileVersion:
    """Cached fit reports are keyed on (profile_version, role_id). Editing
    your profile must invalidate them; reformatting the file must not."""

    def test_is_stable_across_reloads(self) -> None:
        assert profile_version(load_profile(EXAMPLE)) == profile_version(load_profile(EXAMPLE))

    def test_changes_when_a_fact_changes(self, tmp_path: Path) -> None:
        original = profile_version(load_profile(EXAMPLE))
        edited = profile_version(load_profile(write(tmp_path, headline="Staff Engineer")))
        assert original != edited

    def test_ignores_key_order_and_whitespace(self, tmp_path: Path) -> None:
        data = json.loads(EXAMPLE.read_text())
        reordered = tmp_path / "profile.json"
        reordered.write_text(json.dumps(dict(reversed(list(data.items()))), indent=8))
        assert profile_version(load_profile(reordered)) == profile_version(load_profile(EXAMPLE))

    def test_is_short_enough_to_read(self) -> None:
        assert len(profile_version(load_profile(EXAMPLE))) == 16
