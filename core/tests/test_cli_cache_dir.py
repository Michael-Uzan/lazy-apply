"""`--cache-dir` has to move everything derived, not just the vectors.

Found the hard way: the usage log was resolved from `settings.cache_dir` while
every other cached artefact honoured the flag, so a run with `--cache-dir
/tmp/...` still appended to the repository's real log. Anything derived
belongs under one root the flag controls.
"""

from pathlib import Path

from typer.testing import CliRunner

from lazyapply_core.cli import app
from lazyapply_core.config import REPO_ROOT, load_settings

runner = CliRunner()
EXAMPLE = str(REPO_ROOT / "data/example")


def run(command: str, *args: str, cache: Path) -> object:
    return runner.invoke(
        app,
        [command, *args, "--data-dir", EXAMPLE, "--cache-dir", str(cache), "--fake", "--in-memory"],
    )


class TestSettings:
    def test_the_override_lands_on_settings(self, tmp_path: Path) -> None:
        assert load_settings(cache_dir=tmp_path).cache_dir == tmp_path

    def test_the_usage_log_follows_it(self, tmp_path: Path) -> None:
        assert load_settings(cache_dir=tmp_path).usage_log.parent == tmp_path

    def test_the_role_cache_follows_it(self, tmp_path: Path) -> None:
        assert load_settings(cache_dir=tmp_path).roles_cache.parent == tmp_path


class TestCommands:
    def test_ingest_writes_its_usage_log_under_the_override(self, tmp_path: Path) -> None:
        result = run("ingest", cache=tmp_path)
        assert result.exit_code == 0, result.output
        run("ingest", cache=tmp_path)  # second run produces cache hits to log
        assert (tmp_path / "usage.jsonl").is_file()

    def test_nothing_is_written_outside_the_override(self, tmp_path: Path) -> None:
        """The check that would have caught the bug: the repository's own
        cache must be untouched by a run that was pointed elsewhere."""
        before = _snapshot(REPO_ROOT / ".cache")
        run("ingest", cache=tmp_path)
        run("fit", str(REPO_ROOT / "data/example/roles/strong-match.txt"), cache=tmp_path)
        assert _snapshot(REPO_ROOT / ".cache") == before


def _snapshot(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    return {(str(p.relative_to(path)), p.stat().st_size) for p in path.rglob("*") if p.is_file()}
