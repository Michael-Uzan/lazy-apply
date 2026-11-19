"""The `lazyapply embed` command."""

from typer.testing import CliRunner

from lazyapply_core.cli import app
from lazyapply_core.config import REPO_ROOT

runner = CliRunner()
EXAMPLE = str(REPO_ROOT / "data/example")


def run(*args: str, cache: str) -> object:
    return runner.invoke(app, ["embed", "--data-dir", EXAMPLE, "--cache-dir", cache, *args])


class TestDryRun:
    def test_reports_what_would_be_embedded(self, tmp_path) -> None:
        result = run("--dry-run", "--fake", cache=str(tmp_path))
        assert result.exit_code == 0
        assert "6 to embed" in result.output

    def test_reports_an_estimated_cost(self, tmp_path) -> None:
        result = run("--dry-run", "--fake", cache=str(tmp_path))
        assert "estimated cost" in result.output

    def test_writes_nothing(self, tmp_path) -> None:
        """A dry run must not populate the cache it is reporting on."""
        run("--dry-run", "--fake", cache=str(tmp_path))
        assert list(tmp_path.rglob("*.json")) == []


class TestEmbedding:
    def test_embeds_and_caches(self, tmp_path) -> None:
        result = run("--fake", cache=str(tmp_path))
        assert result.exit_code == 0
        assert len(list(tmp_path.rglob("*.json"))) == 6

    def test_second_run_is_a_full_cache_hit(self, tmp_path) -> None:
        run("--fake", cache=str(tmp_path))
        result = run("--fake", cache=str(tmp_path))
        assert "0 to embed" in result.output
        assert "6 cached" in result.output

    def test_dry_run_after_embedding_reports_nothing_to_do(self, tmp_path) -> None:
        run("--fake", cache=str(tmp_path))
        result = run("--dry-run", "--fake", cache=str(tmp_path))
        assert "0 to embed" in result.output
