"""`lazyapply cost` — what has been spent, from the usage log."""

import json
from pathlib import Path

from typer.testing import CliRunner

from lazyapply_core.cli import app

runner = CliRunner()


def log(path: Path, *rows: dict[str, object]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def event(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "command": "fit",
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
        "input_tokens": 2000,
        "output_tokens": 800,
        "latency_ms": 1500,
        "run": "r1",
        "cached": False,
        "at": "2026-11-19T12:00:00+00:00",
        "cost_usd": 0.006,
    }
    row.update(overrides)
    return row


def run(*args: str, cache: Path) -> object:
    return runner.invoke(app, ["cost", *args, "--cache-dir", str(cache)])


class TestOutput:
    def test_reports_the_total(self, tmp_path: Path) -> None:
        log(tmp_path / "usage.jsonl", event(), event(run="r2"))
        result = run(cache=tmp_path)
        assert result.exit_code == 0, result.output
        assert "0.012" in result.output

    def test_breaks_down_by_command(self, tmp_path: Path) -> None:
        log(
            tmp_path / "usage.jsonl",
            event(command="fit"),
            event(command="requirements", model="claude-opus-5", run="r2"),
        )
        out = run(cache=tmp_path).output
        assert "fit" in out
        assert "requirements" in out

    def test_breaks_down_by_model(self, tmp_path: Path) -> None:
        log(tmp_path / "usage.jsonl", event(model="claude-opus-5"))
        assert "claude-opus-5" in run(cache=tmp_path).output

    def test_reports_cost_per_run(self, tmp_path: Path) -> None:
        """The number a user budgets with: what one command costs, not what
        one API call costs."""
        log(tmp_path / "usage.jsonl", event(run="r1"), event(run="r1"), event(run="r2"))
        assert "per run" in run(cache=tmp_path).output.lower()

    def test_reports_what_the_cache_saved(self, tmp_path: Path) -> None:
        log(tmp_path / "usage.jsonl", event(), event(cached=True, cost_usd=0.0, run="r2"))
        assert "saved" in run(cache=tmp_path).output.lower()

    def test_an_empty_log_says_so_rather_than_printing_zeros(self, tmp_path: Path) -> None:
        """A table of zeros reads as "this is free". Nothing has run yet is a
        different statement."""
        out = run(cache=tmp_path).output.lower()
        assert "nothing recorded" in out

    def test_names_the_log_it_read(self, tmp_path: Path) -> None:
        log(tmp_path / "usage.jsonl", event())
        assert "usage.jsonl" in run(cache=tmp_path).output
