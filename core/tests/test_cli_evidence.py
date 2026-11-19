"""The `lazyapply evidence` commands."""

from pathlib import Path

from typer.testing import CliRunner

from lazyapply_core.cli import app
from lazyapply_core.config import REPO_ROOT

runner = CliRunner()
EXAMPLE = str(REPO_ROOT / "data/example")


class TestEvidenceList:
    def test_exits_cleanly(self) -> None:
        result = runner.invoke(app, ["evidence", "list", "--data-dir", EXAMPLE])
        assert result.exit_code == 0

    def test_shows_every_block(self) -> None:
        result = runner.invoke(app, ["evidence", "list", "--data-dir", EXAMPLE])
        assert result.output.count("ev_") >= 6

    def test_shows_readable_ids(self) -> None:
        result = runner.invoke(app, ["evidence", "list", "--data-dir", EXAMPLE])
        assert "ev_northwind__real-time-trading-dashboard" in result.output

    def test_reports_the_total(self) -> None:
        result = runner.invoke(app, ["evidence", "list", "--data-dir", EXAMPLE])
        assert "6 blocks" in result.output


class TestEvidenceShow:
    def test_prints_the_embedded_text_not_the_raw_body(self) -> None:
        """`show` exists to answer 'what did we actually embed?' when a
        retrieval result looks wrong."""
        result = runner.invoke(
            app,
            [
                "evidence",
                "show",
                "ev_northwind__real-time-trading-dashboard",
                "--data-dir",
                EXAMPLE,
            ],
        )
        assert result.exit_code == 0
        assert "Northwind Trading — Real-time trading dashboard" in result.output
        assert "tags: react" in result.output

    def test_unknown_id_fails_with_a_useful_message(self) -> None:
        result = runner.invoke(app, ["evidence", "show", "ev_no__such", "--data-dir", EXAMPLE])
        assert result.exit_code != 0
        assert "ev_no__such" in result.output


class TestErrorHandling:
    def test_parse_errors_are_reported_without_a_traceback(self, tmp_path: Path) -> None:
        """A malformed evidence file is a user mistake, not a crash."""
        (tmp_path / "evidence").mkdir()
        (tmp_path / "evidence" / "bad.md").write_text("## No front matter\nProse.\n")
        result = runner.invoke(app, ["evidence", "list", "--data-dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "front matter" in result.output
        assert "Traceback" not in result.output
