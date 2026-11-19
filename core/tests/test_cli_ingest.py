"""`lazyapply ingest` and `lazyapply search`.

Run with --fake (no key) and --in-memory (no Qdrant), so the whole command
path is covered offline.
"""

from typer.testing import CliRunner

from lazyapply_core.cli import app
from lazyapply_core.config import REPO_ROOT

runner = CliRunner()
EXAMPLE = str(REPO_ROOT / "data/example")


def run(*args: str, cache: str) -> object:
    return runner.invoke(
        app, [*args, "--data-dir", EXAMPLE, "--cache-dir", cache, "--fake", "--in-memory"]
    )


class TestIngest:
    def test_succeeds(self, tmp_path) -> None:
        result = run("ingest", cache=str(tmp_path))
        assert result.exit_code == 0

    def test_reports_counts(self, tmp_path) -> None:
        result = run("ingest", cache=str(tmp_path))
        assert "6 blocks" in result.output
        assert "6 embedded" in result.output

    def test_second_run_embeds_nothing(self, tmp_path) -> None:
        run("ingest", cache=str(tmp_path))
        result = run("ingest", cache=str(tmp_path))
        assert "0 embedded" in result.output
        assert "6 cached" in result.output


class TestSearch:
    def test_returns_scored_hits(self, tmp_path) -> None:
        """--in-memory means ingest and search share one process, so search
        runs against what ingest just stored."""
        result = run("search", "trading dashboard", cache=str(tmp_path))
        assert result.exit_code == 0
        assert "ev_" in result.output

    def test_respects_limit(self, tmp_path) -> None:
        result = run("search", "anything", "--limit", "2", cache=str(tmp_path))
        assert result.output.count("ev_") == 2

    def test_reports_scores(self, tmp_path) -> None:
        result = run("search", "anything", "--limit", "1", cache=str(tmp_path))
        assert "." in result.output


class TestProviderFailureHandling:
    """A rate limit is a message, not a stack trace."""

    def test_embedding_errors_print_one_line(self, tmp_path, monkeypatch) -> None:
        from lazyapply_core.embed.base import EmbeddingError

        def explode(*args: object, **kwargs: object) -> None:
            raise EmbeddingError("voyage-3-lite: rate limit of 3 RPM exceeded")

        monkeypatch.setattr("lazyapply_core.cli.ingest.build_embedder", explode)
        result = runner.invoke(
            app, ["search", "anything", "--data-dir", EXAMPLE, "--cache-dir", str(tmp_path)]
        )
        assert result.exit_code != 0
        assert "rate limit" in result.output
        assert "Traceback" not in result.output
