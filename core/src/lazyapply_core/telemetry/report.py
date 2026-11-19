"""Aggregating the usage log into a report.

Pure arithmetic over recorded events. Three grouping decisions, each chosen
around a question someone actually asks:

- **Per command, per run.** "What does a fit check cost" -- not what a
  ``messages.parse`` costs. A check makes two provider calls, so runs are
  counted rather than calls.
- **Median latency, not mean.** One rate-limited retry on a 3 RPM free tier
  took two minutes. A mean would let that define what the command feels like.
- **Ordered by spend.** The expensive command is the one worth looking at.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from statistics import median

from lazyapply_core.telemetry.base import Usage
from lazyapply_core.telemetry.pricing import cost_usd


@dataclass(frozen=True)
class Group:
    """One command or model, summed."""

    key: str
    calls: int
    runs: int
    input_tokens: int
    output_tokens: int
    total_usd: float

    median_latency_ms: int
    """Median of one call. The right number for a model."""

    median_run_ms: int
    """Median of a whole invocation -- its calls summed, then the median over
    runs. The right number for a command: a fit check's 459ms embedding and
    8.9s judgment describe the check only when added together."""

    @property
    def command(self) -> str:
        return self.key

    @property
    def model(self) -> str:
        return self.key

    @property
    def usd_per_run(self) -> float:
        """The number a user budgets with."""
        return self.total_usd / self.runs if self.runs else 0.0


@dataclass(frozen=True)
class Summary:
    calls: int
    cached_calls: int
    input_tokens: int
    output_tokens: int
    total_usd: float

    saved_usd: float
    """What the caches avoided spending. The only place that is visible -- a
    report of money spent says nothing about money not spent."""

    by_command: tuple[Group, ...]
    by_model: tuple[Group, ...]
    first: datetime | None
    last: datetime | None


def summarise(events: Sequence[Usage]) -> Summary:
    """Turn a usage log into a report."""
    if not events:
        return Summary(0, 0, 0, 0, 0.0, 0.0, (), (), None, None)

    billed = [e for e in events if not e.cached]
    cached = [e for e in events if e.cached]

    return Summary(
        calls=len(events),
        cached_calls=len(cached),
        input_tokens=sum(e.input_tokens for e in events),
        output_tokens=sum(e.output_tokens for e in events),
        total_usd=sum(e.cost_usd for e in events),
        saved_usd=sum(_list_price(e) for e in cached),
        by_command=_group(billed or events, lambda e: e.command),
        by_model=_group(billed or events, lambda e: e.model),
        first=min(e.at for e in events),
        last=max(e.at for e in events),
    )


def _list_price(event: Usage) -> float:
    """What a cache hit would have cost had it been a real request."""
    return cost_usd(event.model, input_tokens=event.input_tokens, output_tokens=event.output_tokens)


def _median_run_ms(events: Sequence[Usage]) -> int:
    """Sum each run's calls, then take the median across runs."""
    per_run: dict[str, int] = {}
    for event in events:
        per_run[event.run] = per_run.get(event.run, 0) + event.latency_ms
    return int(median(per_run.values())) if per_run else 0


def _group(events: Sequence[Usage], key: Callable[[Usage], str]) -> tuple[Group, ...]:
    buckets: dict[str, list[Usage]] = {}
    for event in events:
        buckets.setdefault(key(event), []).append(event)

    groups = [
        Group(
            key=name,
            calls=len(bucket),
            runs=len({e.run for e in bucket}),
            input_tokens=sum(e.input_tokens for e in bucket),
            output_tokens=sum(e.output_tokens for e in bucket),
            total_usd=sum(e.cost_usd for e in bucket),
            median_latency_ms=int(median(e.latency_ms for e in bucket)),
            median_run_ms=_median_run_ms(bucket),
        )
        for name, bucket in buckets.items()
    ]
    return tuple(sorted(groups, key=lambda g: g.total_usd, reverse=True))
