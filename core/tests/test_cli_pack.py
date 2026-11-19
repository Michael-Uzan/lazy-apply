"""`lazyapply pack <role file>` — inspect what matching will be given."""

from pathlib import Path

from typer.testing import CliRunner

from lazyapply_core.cli import app
from lazyapply_core.config import REPO_ROOT

runner = CliRunner()
EXAMPLE = str(REPO_ROOT / "data/example")
ROLE = str(REPO_ROOT / "data/example/roles/strong-match.txt")


def run(*args: str, cache: Path) -> object:
    return runner.invoke(
        app,
        [
            "pack",
            ROLE,
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
        assert run(cache=tmp_path).exit_code == 0

    def test_lists_every_requirement(self, tmp_path: Path) -> None:
        """The canned extractor returns nine requirements."""
        out = run(cache=tmp_path).output
        assert out.count("must-have") + out.count("nice-to-have") >= 9

    def test_shows_retrieved_evidence_ids_with_scores(self, tmp_path: Path) -> None:
        out = run(cache=tmp_path).output
        assert "ev_" in out
        assert "0." in out

    def test_reports_how_many_blocks_will_be_sent(self, tmp_path: Path) -> None:
        """The same block is retrieved for several requirements but is sent
        once, so the deduplicated count is the one that matters for cost."""
        assert "unique blocks" in run(cache=tmp_path).output

    def test_respects_top_k(self, tmp_path: Path) -> None:
        few = run("--top-k", "1", cache=tmp_path).output
        many = run("--top-k", "3", cache=tmp_path).output
        assert many.count("ev_") > few.count("ev_")


class TestErrors:
    def test_missing_role_file_fails_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "pack",
                str(tmp_path / "nope.txt"),
                "--data-dir",
                EXAMPLE,
                "--cache-dir",
                str(tmp_path),
                "--fake",
                "--in-memory",
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output
        assert "Traceback" not in result.output
