"""`lazyapply fit <role file>` — the Phase 1 deliverable.

Runs the whole pipeline with the fake embedder, an in-process store and the
offline score-based evaluator: no key, no Qdrant, no cost. What is being
checked is that the command reports a verdict, cites evidence, names gaps, and
never shows the user a traceback.
"""

from pathlib import Path

from typer.testing import CliRunner

from lazyapply_core.cli import app
from lazyapply_core.config import REPO_ROOT

runner = CliRunner()
EXAMPLE = str(REPO_ROOT / "data/example")
STRONG = str(REPO_ROOT / "data/example/roles/strong-match.txt")


def run(*args: str, cache: Path, role: str = STRONG) -> object:
    return runner.invoke(
        app,
        [
            "fit",
            role,
            *args,
            "--data-dir",
            EXAMPLE,
            "--cache-dir",
            str(cache),
            "--fake",
            "--in-memory",
        ],
    )


class TestOutput:
    def test_exits_cleanly(self, tmp_path: Path) -> None:
        result = run(cache=tmp_path)
        assert result.exit_code == 0, result.output

    def test_reports_a_verdict(self, tmp_path: Path) -> None:
        out = run(cache=tmp_path).output.lower()
        assert any(v in out for v in ("generate", "borderline", "skip"))

    def test_reports_coverage_and_score(self, tmp_path: Path) -> None:
        out = run(cache=tmp_path).output.lower()
        assert "coverage" in out
        assert "score" in out

    def test_shows_a_status_per_requirement(self, tmp_path: Path) -> None:
        out = run(cache=tmp_path).output.lower()
        assert any(s in out for s in ("strong", "partial", "none"))

    def test_cites_evidence_ids(self, tmp_path: Path) -> None:
        """A verdict a reader cannot audit is worth nothing."""
        assert "ev_" in run(cache=tmp_path).output

    def test_shows_rationales(self, tmp_path: Path) -> None:
        assert "fake evaluator" in run(cache=tmp_path).output

    def test_writes_json_when_asked(self, tmp_path: Path) -> None:
        """Phase 2's CV agent consumes the report, not the printed text."""
        out = tmp_path / "report.json"
        result = run("--json-out", str(out), cache=tmp_path)
        assert result.exit_code == 0, result.output
        assert '"verdict"' in out.read_text()
        assert '"must_have_coverage"' in out.read_text()


class TestErrors:
    def test_a_missing_role_file_fails_cleanly(self, tmp_path: Path) -> None:
        result = run(cache=tmp_path, role=str(tmp_path / "nope.txt"))
        assert result.exit_code != 0
        assert "not found" in result.output
        assert "Traceback" not in result.output

    def test_a_missing_profile_fails_cleanly(self, tmp_path: Path) -> None:
        """Hard facts are not optional -- without them education requirements
        cannot be judged at all."""
        result = runner.invoke(
            app,
            [
                "fit",
                STRONG,
                "--data-dir",
                str(tmp_path),
                "--cache-dir",
                str(tmp_path),
                "--fake",
                "--in-memory",
            ],
        )
        assert result.exit_code != 0
        assert "Traceback" not in result.output
