"""Recording what each call cost.

The recorder is a seam like the embedder and the store: a protocol with a
no-op default, so nothing in the pipeline has to know whether telemetry is
switched on. The log is append-only JSONL -- one line per call, never
rewritten -- because the only question it has to answer is "what did this
cost", and a format you can `tail` beats one you have to parse.
"""

import json
from pathlib import Path

import pytest

from lazyapply_core.telemetry.base import NullTelemetry, Usage
from lazyapply_core.telemetry.jsonl import JsonlTelemetry, read_usage


def usage(**overrides: object) -> Usage:
    fields: dict[str, object] = {
        "command": "fit",
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
        "input_tokens": 2000,
        "output_tokens": 800,
        "latency_ms": 1500,
    }
    fields.update(overrides)
    return Usage(**fields)  # type: ignore[arg-type]


class TestUsage:
    def test_computes_its_own_cost(self) -> None:
        assert usage().cost_usd == pytest.approx(0.006)

    def test_a_cached_call_costs_nothing(self) -> None:
        """A cache hit makes no request. Charging for it would make the cache
        look worthless in the one report meant to prove its value."""
        assert usage(cached=True, input_tokens=2000, output_tokens=800).cost_usd == 0.0

    def test_records_when_it_happened(self) -> None:
        assert usage().at.tzinfo is not None


class TestNullTelemetry:
    def test_accepts_events_and_does_nothing(self) -> None:
        """The default. Every provider takes a recorder, so the no-op has to
        be a real object rather than a None check at each call site."""
        NullTelemetry().record(usage())


class TestJsonlTelemetry:
    def test_writes_one_line_per_call(self, tmp_path: Path) -> None:
        log = tmp_path / "usage.jsonl"
        telemetry = JsonlTelemetry(log)
        telemetry.record(usage())
        telemetry.record(usage(command="eval"))

        assert len(log.read_text().strip().splitlines()) == 2

    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        log = tmp_path / "usage.jsonl"
        JsonlTelemetry(log).record(usage())
        assert json.loads(log.read_text().strip())["model"] == "claude-haiku-4-5"

    def test_records_the_computed_cost(self, tmp_path: Path) -> None:
        """Written at record time, not derived on read -- a report of what a
        run cost must not change when a price does."""
        log = tmp_path / "usage.jsonl"
        JsonlTelemetry(log).record(usage())
        assert json.loads(log.read_text())["cost_usd"] == pytest.approx(0.006)

    def test_appends_rather_than_overwriting(self, tmp_path: Path) -> None:
        log = tmp_path / "usage.jsonl"
        JsonlTelemetry(log).record(usage())
        JsonlTelemetry(log).record(usage())
        assert len(log.read_text().strip().splitlines()) == 2

    def test_creates_the_directory(self, tmp_path: Path) -> None:
        log = tmp_path / "nested" / "deeper" / "usage.jsonl"
        JsonlTelemetry(log).record(usage())
        assert log.is_file()

    def test_a_write_failure_never_breaks_the_command(self, tmp_path: Path) -> None:
        """Telemetry is an observer. Losing a cost line is a nuisance;
        failing a fit check the user asked for is not acceptable."""
        unwritable = tmp_path / "usage.jsonl"
        unwritable.mkdir()
        JsonlTelemetry(unwritable).record(usage())


class TestReadUsage:
    def test_reads_back_what_was_written(self, tmp_path: Path) -> None:
        log = tmp_path / "usage.jsonl"
        telemetry = JsonlTelemetry(log)
        telemetry.record(usage())
        telemetry.record(usage(command="eval", model="claude-opus-5"))

        events = read_usage(log)
        assert [e.command for e in events] == ["fit", "eval"]

    def test_a_missing_log_reads_as_empty(self, tmp_path: Path) -> None:
        """Nobody has run anything yet. That is not an error."""
        assert read_usage(tmp_path / "nothing.jsonl") == ()

    def test_a_corrupt_line_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        """An interrupted write should cost one line, not the whole history."""
        log = tmp_path / "usage.jsonl"
        JsonlTelemetry(log).record(usage())
        with log.open("a") as handle:
            handle.write("{half a line\n")
        JsonlTelemetry(log).record(usage(command="eval"))

        assert [e.command for e in read_usage(log)] == ["fit", "eval"]


class TestRun:
    def test_carries_the_recorder_command_and_id(self) -> None:
        from lazyapply_core.telemetry import NullTelemetry as Null
        from lazyapply_core.telemetry import Run

        run = Run(telemetry=Null(), command="fit")
        assert run.command == "fit"
        assert run.id

    def test_each_run_gets_its_own_id(self) -> None:
        from lazyapply_core.telemetry import NullTelemetry as Null
        from lazyapply_core.telemetry import Run

        assert Run(telemetry=Null(), command="fit").id != Run(telemetry=Null(), command="fit").id

    def test_a_disabled_run_records_nothing(self, tmp_path: Path) -> None:
        """--no-telemetry has to reach every provider, so it is expressed as
        a run whose recorder discards rather than a flag each one checks."""
        from lazyapply_core.config import load_settings
        from lazyapply_core.telemetry import Run

        settings = load_settings().model_copy(update={"cache_dir": tmp_path})
        run = Run.for_command("fit", settings, enabled=False)
        run.telemetry.record(usage())

        assert not settings.usage_log.exists()

    def test_an_enabled_run_writes_to_the_usage_log(self, tmp_path: Path) -> None:
        from lazyapply_core.config import load_settings
        from lazyapply_core.telemetry import Run

        settings = load_settings().model_copy(update={"cache_dir": tmp_path})
        run = Run.for_command("fit", settings)
        run.telemetry.record(usage())

        assert settings.usage_log.is_file()
