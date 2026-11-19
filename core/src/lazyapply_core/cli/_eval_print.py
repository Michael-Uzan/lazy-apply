"""Printing the eval results.

Split from ``cli/eval.py`` so that file stays about orchestration. Two
presentation decisions carry meaning rather than taste:

- An undefined metric prints as a dash, never 0.000. "We measured nothing"
  and "we measured a complete failure" are different results and must not
  share a glyph.
- Every imperfect retrieval is listed individually. An average of 0.955 is
  only useful if you can see which requirement cost the 0.045.
"""

from __future__ import annotations

import typer

from lazyapply_core.config import Settings
from lazyapply_core.evals.runner import MatchingScore, RetrievalScore, ThresholdMargin

_BORDERLINE_MARGIN = 0.1
"""How close a posting must land to a boundary before it says anything about
where that boundary belongs."""

_NOT_MEASURED = "—"
"""Printed where a metric is undefined. Distinct from 0.000, which is a
measured failure rather than an absent measurement."""


def _number(value: float | None) -> str:
    return _NOT_MEASURED if value is None else f"{value:.3f}"


def print_retrieval(scores: list[RetrievalScore]) -> None:
    """One row per posting per cutoff.

    A sweep reports every k from one pack, so the rows share an embedding
    request and the column has to say which cutoff each row is.
    """
    typer.secho("Retrieval", bold=True)
    typer.echo(f"{'posting':<22}{'k':>3}{'scored':>8}{'recall@k':>11}{'MRR':>8}")

    for score in scores:
        typer.echo(
            f"{score.role:<22}{score.k:>3}{score.scored:>8}"
            f"{_number(score.recall):>11}{_number(score.mrr):>8}"
        )

    for role, count in _excluded(scores).items():
        typer.echo(f"\n{role}: {count} requirement(s) excluded — no relevant evidence to find")

    _print_imperfect(scores)


def _excluded(scores: list[RetrievalScore]) -> dict[str, int]:
    """Requirements with no correct answer, once per posting rather than once
    per cutoff -- the count does not change with k."""
    return {s.role: s.unscoreable for s in scores if s.unscoreable}


def _print_imperfect(scores: list[RetrievalScore]) -> None:
    """Name the requirements that cost the average, at the widest cutoff.

    An average of 0.955 is only useful if you can see which requirement lost
    the 0.045.
    """
    if not scores:
        return

    widest = max(s.k for s in scores)
    imperfect = [d for s in scores if s.k == widest for d in s.details if not d.is_perfect]

    if not imperfect:
        typer.echo(f"\nEvery labelled requirement found its evidence at rank 1 (k={widest}).")
        return

    typer.echo(f"\nLess than perfect at k={widest}:")
    for detail in imperfect:
        where = "not retrieved" if detail.rank is None else f"rank {detail.rank}"
        missing = f", missed {', '.join(detail.missing)}" if detail.missing else ""
        typer.echo(f"  {detail.requirement_id:<28}{where}, recall {detail.recall:.2f}{missing}")


def print_margins(margins: list[ThresholdMargin], settings: Settings) -> None:
    """Report what the thresholds are actually being asked to decide.

    Derived from the hand labels, so it measures the thresholds rather than
    the model. A posting far from every boundary confirms nothing about where
    the boundary belongs -- and printing the distance makes an under-powered
    golden set visible instead of leaving it to be assumed.
    """
    thresholds = settings.thresholds
    typer.echo("")
    typer.secho(
        f"Thresholds  (skip below {thresholds.skip_below}, "
        f"generate at {thresholds.generate_at} with score >= {thresholds.min_score})",
        bold=True,
    )
    typer.echo(f"{'posting':<22}{'coverage':>9}{'score':>7}{'verdict':>12}{'to boundary':>14}")

    for margin in margins:
        typer.echo(
            f"{margin.role:<22}{margin.coverage:>9.3f}{margin.score:>7}"
            f"{margin.verdict.value:>12}{margin.distance_to_boundary:>14.3f}"
        )

    nearest = min((m.distance_to_boundary for m in margins), default=1.0)
    if nearest > _BORDERLINE_MARGIN:
        typer.echo(
            f"\nNo posting lands within {_BORDERLINE_MARGIN} of a boundary, so this set "
            "cannot\ntell whether the thresholds are right. Tuning them needs a posting "
            "that is\ngenuinely borderline."
        )


def print_matching(scores: list[MatchingScore], model: str) -> None:
    typer.echo("")
    typer.secho(f"Matching  ({model})", bold=True)

    for score in scores:
        result = score.agreement
        verdict = f"{score.actual_verdict.value} (expected {score.expected_verdict.value})"
        mark = "ok" if score.verdict_matched else "MISMATCH"
        typer.echo(
            f"{score.role:<22}agreement {result.matched}/{result.total}"
            f"   verdict {verdict} — {mark}"
        )

        for pair in result.disagreements:
            expected, actual = pair
            typer.echo(
                f"{'':<22}expected {expected.value}, got {actual.value} ({result.direction(pair)})"
            )
