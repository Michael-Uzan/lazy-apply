"""`lazyapply cost` — what has actually been spent.

Reads the usage log and reports it the way a user asks about it: per command,
per run. "What does a fit check cost" is answerable; "what does a
``messages.parse`` cost" is not a question anyone has.

Every figure comes from a provider's own token count, recorded at the time of
the call. Nothing here estimates.
"""

from __future__ import annotations

import typer

from lazyapply_core.cli._shared import CacheDir, DataDir, settings_for
from lazyapply_core.telemetry import Summary, read_usage, summarise
from lazyapply_core.telemetry.report import Group


def cost(data_dir: DataDir = None, cache_dir: CacheDir = None) -> None:
    """Report spend from the usage log."""
    settings = settings_for(data_dir, cache_dir)
    summary = summarise(read_usage(settings.usage_log))

    if summary.calls == 0:
        # A table of zeros reads as "this is free". Nothing has run yet is a
        # different statement, and the path is the useful part of it.
        typer.echo(f"nothing recorded yet in {settings.usage_log}")
        return

    _print_totals(summary, settings.usage_log)
    _print_group("by command", summary.by_command, label="command", per_run_latency=True)
    _print_group("by model", summary.by_model, label="model", per_run_latency=False)


def _print_totals(summary: Summary, log: str | object) -> None:
    typer.secho(f"${summary.total_usd:.4f} total", bold=True)
    typer.echo(
        f"{summary.calls} call(s), {summary.input_tokens:,} in / {summary.output_tokens:,} out"
    )

    if summary.cached_calls:
        typer.echo(f"{summary.cached_calls} served from cache — ${summary.saved_usd:.4f} saved")

    if summary.first and summary.last:
        typer.echo(f"{summary.first:%Y-%m-%d} to {summary.last:%Y-%m-%d}   ({log})")


def _print_group(
    title: str, groups: tuple[Group, ...], *, label: str, per_run_latency: bool
) -> None:
    """One table. ``per_run_latency`` picks which latency the column means:
    a whole invocation for a command, a single call for a model."""
    if not groups:
        return

    heading = "median run" if per_run_latency else "median call"
    typer.echo("")
    typer.secho(title, bold=True)
    typer.echo(f"{label:<16}{'runs':>6}{'total':>10}{'per run':>10}{heading:>13}")

    for group in groups:
        latency = group.median_run_ms if per_run_latency else group.median_latency_ms
        typer.echo(
            f"{group.key:<16}{group.runs:>6}"
            f"{'$' + format(group.total_usd, '.4f'):>10}"
            f"{'$' + format(group.usd_per_run, '.4f'):>10}"
            f"{format(latency, ',') + 'ms':>13}"
        )
