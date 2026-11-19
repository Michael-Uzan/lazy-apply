"""`lazyapply requirements <role file>`."""

from pathlib import Path

from typer.testing import CliRunner

from lazyapply_core.cli import app
from lazyapply_core.config import REPO_ROOT

runner = CliRunner()
ROLE = str(REPO_ROOT / "data/example/roles/strong-match.txt")


def run(*args: str, cache: Path) -> object:
    return runner.invoke(app, ["requirements", *args, "--cache-dir", str(cache), "--fake"])


class TestOutput:
    def test_exits_cleanly(self, tmp_path: Path) -> None:
        assert run(ROLE, cache=tmp_path).exit_code == 0

    def test_prints_the_title(self, tmp_path: Path) -> None:
        assert "Senior Full-Stack Engineer" in run(ROLE, cache=tmp_path).output

    def test_separates_must_haves_from_nice_to_haves(self, tmp_path: Path) -> None:
        out = run(ROLE, cache=tmp_path).output
        assert "must-have" in out.lower()
        assert "nice-to-have" in out.lower()

    def test_prints_the_role_id(self, tmp_path: Path) -> None:
        """The role id is the extraction cache key; showing it makes a stale
        cache diagnosable."""
        from lazyapply_core.hashing import content_hash

        expected = content_hash(Path(ROLE).read_text())
        assert expected in run(ROLE, cache=tmp_path).output


class TestErrors:
    def test_missing_file_fails_without_a_traceback(self, tmp_path: Path) -> None:
        result = run(str(tmp_path / "nope.txt"), cache=tmp_path)
        assert result.exit_code != 0
        assert "not found" in result.output
        assert "Traceback" not in result.output

    def test_empty_file_is_rejected(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.txt"
        empty.write_text("   \n")
        result = run(str(empty), cache=tmp_path)
        assert result.exit_code != 0
        assert "empty" in result.output
