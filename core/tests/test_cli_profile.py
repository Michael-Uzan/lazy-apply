"""`lazyapply profile show`."""

import json
from pathlib import Path

from typer.testing import CliRunner

from lazyapply_core.cli import app
from lazyapply_core.config import REPO_ROOT

runner = CliRunner()
EXAMPLE = str(REPO_ROOT / "data/example")


def show(data_dir: str = EXAMPLE) -> object:
    return runner.invoke(app, ["profile", "show", "--data-dir", data_dir])


class TestShow:
    def test_exits_cleanly(self) -> None:
        assert show().exit_code == 0

    def test_prints_identity_and_contact(self) -> None:
        out = show().output
        assert "Dana Levi" in out
        assert "dana.levi@example.com" in out

    def test_prints_education_and_positions(self) -> None:
        out = show().output
        assert "Technion" in out
        assert "Northwind Trading" in out
        assert "present" in out

    def test_prints_the_profile_version(self) -> None:
        """The version is what invalidates cached fit reports, so it must be
        visible when a cache behaves unexpectedly."""
        from lazyapply_core.ingest.profile import load_profile, profile_version

        expected = profile_version(load_profile(REPO_ROOT / "data/example/profile.json"))
        assert expected in show().output


class TestErrors:
    def test_missing_profile_fails_without_a_traceback(self, tmp_path: Path) -> None:
        result = show(str(tmp_path))
        assert result.exit_code != 0
        assert "profile not found" in result.output
        assert "Traceback" not in result.output

    def test_invalid_profile_names_the_field(self, tmp_path: Path) -> None:
        data = json.loads(Path(REPO_ROOT / "data/example/profile.json").read_text())
        data["email"] = "nope"
        (tmp_path / "profile.json").write_text(json.dumps(data))
        result = show(str(tmp_path))
        assert result.exit_code != 0
        assert "email" in result.output
