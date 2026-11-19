"""Aggregating the usage log.

Pure arithmetic over recorded events. The grouping is chosen around the
question a user actually asks -- "what does one fit check cost me" -- which
means per-command averages, not a single running total.
"""

from datetime import UTC, datetime, timedelta

import pytest

from lazyapply_core.telemetry.base import Usage
from lazyapply_core.telemetry.report import summarise

NOW = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)


def usage(command: str = "fit", model: str = "claude-haiku-4-5", **kw: object) -> Usage:
    fields: dict[str, object] = {
        "command": command,
        "provider": "anthropic",
        "model": model,
        "input_tokens": 2000,
        "output_tokens": 800,
        "latency_ms": 1500,
        "at": NOW,
    }
    fields.update(kw)
    return Usage(**fields)  # type: ignore[arg-type]


class TestTotals:
    def test_an_empty_log_summarises_to_nothing(self) -> None:
        summary = summarise(())
        assert summary.total_usd == 0.0
        assert summary.calls == 0
        assert summary.by_command == ()

    def test_sums_the_cost(self) -> None:
        assert summarise((usage(), usage())).total_usd == pytest.approx(0.012)

    def test_counts_the_calls(self) -> None:
        assert summarise((usage(), usage(), usage())).calls == 3

    def test_sums_the_tokens(self) -> None:
        summary = summarise((usage(), usage()))
        assert summary.input_tokens == 4000
        assert summary.output_tokens == 1600


class TestPerCommand:
    def test_groups_by_command(self) -> None:
        summary = summarise((usage("fit"), usage("fit"), usage("eval")))
        assert {row.command: row.calls for row in summary.by_command} == {"fit": 2, "eval": 1}

    def test_reports_the_cost_per_run_not_per_call(self) -> None:
        """A fit check makes two provider calls -- one embedding, one
        judgment. The number a user wants is what the command costs, so runs
        are counted rather than calls."""
        summary = summarise(
            (
                usage("fit", model="voyage-3-lite", run="r1"),
                usage("fit", model="claude-haiku-4-5", run="r1"),
                usage("fit", model="voyage-3-lite", run="r2"),
                usage("fit", model="claude-haiku-4-5", run="r2"),
            )
        )
        row = summary.by_command[0]
        assert row.runs == 2
        assert row.usd_per_run == pytest.approx(row.total_usd / 2)

    def test_orders_by_spend(self) -> None:
        """The expensive command is the one worth looking at first."""
        summary = summarise(
            (usage("fit"), usage("requirements", model="claude-opus-5"), usage("fit"))
        )
        assert [row.command for row in summary.by_command] == ["requirements", "fit"]

    def test_reports_median_latency(self) -> None:
        """Median rather than mean: one rate-limited retry should not define
        what the command normally feels like."""
        summary = summarise(
            (
                usage(latency_ms=100),
                usage(latency_ms=200),
                usage(latency_ms=90_000),
            )
        )
        assert summary.by_command[0].median_latency_ms == 200


class TestPerModel:
    def test_groups_by_model(self) -> None:
        summary = summarise((usage(model="claude-opus-5"), usage(model="claude-haiku-4-5")))
        assert {row.model for row in summary.by_model} == {"claude-opus-5", "claude-haiku-4-5"}

    def test_orders_by_spend(self) -> None:
        summary = summarise(
            (
                usage(model="voyage-3-lite"),
                usage(model="claude-opus-5"),
                usage(model="voyage-3-lite"),
            )
        )
        assert summary.by_model[0].model == "claude-opus-5"


class TestCaching:
    def test_counts_cache_hits_separately(self) -> None:
        summary = summarise((usage(), usage(cached=True), usage(cached=True)))
        assert summary.cached_calls == 2
        assert summary.calls == 3

    def test_a_cache_hit_adds_no_cost(self) -> None:
        assert summarise((usage(cached=True),)).total_usd == 0.0

    def test_reports_what_the_cache_saved(self) -> None:
        """The only place the cache's value is visible. Without this the
        report shows money spent and says nothing about money not spent."""
        summary = summarise((usage(), usage(cached=True)))
        assert summary.saved_usd == pytest.approx(0.006)


class TestWindow:
    def test_reports_the_period_covered(self) -> None:
        summary = summarise((usage(at=NOW), usage(at=NOW + timedelta(days=2))))
        assert summary.first is not None
        assert summary.last is not None
        assert (summary.last - summary.first).days == 2


class TestLatency:
    """Two different questions, two different numbers.

    A command's latency is how long the whole invocation took -- the sum of
    its calls. A model's latency is how long one call to it took. Reporting a
    command's as the median across its calls mixes a 459ms embedding with an
    8.9s judgment and produces a number describing neither.
    """

    def test_a_commands_latency_is_the_sum_of_its_calls(self) -> None:
        summary = summarise(
            (
                usage(model="voyage-3-lite", latency_ms=459, run="r1"),
                usage(model="claude-haiku-4-5", latency_ms=8890, run="r1"),
            )
        )
        assert summary.by_command[0].median_run_ms == 9349

    def test_several_runs_take_the_median_of_their_totals(self) -> None:
        summary = summarise(
            (
                usage(latency_ms=1000, run="r1"),
                usage(latency_ms=2000, run="r2"),
                usage(latency_ms=9000, run="r3"),
            )
        )
        assert summary.by_command[0].median_run_ms == 2000

    def test_a_models_latency_stays_per_call(self) -> None:
        summary = summarise(
            (
                usage(model="claude-haiku-4-5", latency_ms=8000, run="r1"),
                usage(model="claude-haiku-4-5", latency_ms=9000, run="r2"),
            )
        )
        assert summary.by_model[0].median_latency_ms == 8500
