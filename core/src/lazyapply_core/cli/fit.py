"""`lazyapply fit` — the Phase 1 deliverable.

Runs the whole pipeline: extract requirements, retrieve per requirement, judge
against the evidence and the hard facts, verify the citations, compute the
verdict. Everything printed here is auditable -- every status carries the ids
it was based on, and the verdict is arithmetic over the statuses rather than
an opinion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from lazyapply_core.cli._shared import (
    CacheDir,
    DataDir,
    Fake,
    InMemory,
    NoTelemetry,
    abort,
    run_for,
    settings_for,
)
from lazyapply_core.config import Settings
from lazyapply_core.domain import FitReport, RequirementKind, Role, Verdict
from lazyapply_core.embed import build_embedder
from lazyapply_core.embed.base import EmbeddingError
from lazyapply_core.evaluate import CitationError, EvaluationError, build_evaluator, evaluate_fit
from lazyapply_core.ingest.parser import ParseError
from lazyapply_core.ingest.pipeline import ingest as run_ingest
from lazyapply_core.ingest.profile import load_profile, profile_version
from lazyapply_core.retrieve import ExtractionError, build_extractor, extract_role
from lazyapply_core.retrieve.retriever import build_pack
from lazyapply_core.store import build_store
from lazyapply_core.telemetry import Run

RoleFile = Annotated[Path, typer.Argument(help="Plain-text job posting.")]
TopK = Annotated[int, typer.Option("--top-k", help="Candidates retrieved per requirement.")]
JsonOut = Annotated[
    Path | None,
    typer.Option("--json-out", help="Also write the report as JSON, for Phase 2's CV agent."),
]

_STATUS_WIDTH = 8
"""Pads the status column so requirement ids line up beneath each other."""

_INDENT = " " * 10
"""Aligns rationales and citations under their requirement."""

_COLOUR = {
    Verdict.GENERATE: typer.colors.GREEN,
    Verdict.BORDERLINE: typer.colors.YELLOW,
    Verdict.SKIP: typer.colors.RED,
}


def fit(
    role_file: RoleFile,
    top_k: TopK = 3,
    json_out: JsonOut = None,
    data_dir: DataDir = None,
    cache_dir: CacheDir = None,
    fake: Fake = False,
    in_memory: InMemory = False,
    no_telemetry: NoTelemetry = False,
) -> None:
    """Decide whether the profile fits a job posting, with citations."""
    settings = settings_for(data_dir, cache_dir)
    run = run_for("fit", settings, no_telemetry=no_telemetry)

    if not role_file.is_file():
        raise abort(f"role file not found: {role_file}")

    try:
        role, report = _pipeline(
            role_file,
            settings,
            top_k=top_k,
            fake=fake,
            in_memory=in_memory,
            run=run,
        )
    except (
        ParseError,
        ExtractionError,
        EmbeddingError,
        EvaluationError,
        CitationError,
        ValueError,
    ) as exc:
        raise abort(str(exc)) from exc

    _print(role, report)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(report.model_dump_json(indent=2))
        typer.echo(f"\nwrote {json_out}")


def _pipeline(
    role_file: Path,
    settings: Settings,
    *,
    top_k: int,
    fake: bool,
    in_memory: bool,
    run: Run,
) -> tuple[Role, FitReport]:

    profile = load_profile(settings.profile_path)
    embedder = build_embedder(settings, fake=fake, run=run)
    store = build_store(settings, in_memory=in_memory)
    if in_memory and store.count() == 0:
        run_ingest(settings, embedder=embedder, store=store)

    role = extract_role(
        role_file.read_text(),
        extractor=build_extractor(settings, fake=fake, run=run),
        cache_dir=settings.roles_cache,
    )
    pack = build_pack(role, embedder=embedder, store=store, top_k=top_k)

    report = evaluate_fit(
        role,
        pack,
        profile,
        evaluator=build_evaluator(settings, fake=fake, run=run),
        thresholds=settings.thresholds,
        profile_version=profile_version(profile),
    )
    return role, report


def _print(role: Role, report: FitReport) -> None:
    header = role.title if not role.company else f"{role.title} — {role.company}"
    typer.echo(f"{header}\nrole_id   {report.role_id}\nprofile   {report.profile_version}\n")

    typer.secho(f"verdict   {report.verdict.value.upper()}", fg=_COLOUR[report.verdict], bold=True)
    typer.echo(f"coverage  {report.must_have_coverage:.0%} of must-haves (partials count half)")
    typer.echo(f"score     {report.score}/100\n")

    kinds = {r.id: r.kind for r in role.requirements}
    for match in report.matches:
        kind = kinds[match.requirement_id]
        typer.echo(f"{match.status.value:<{_STATUS_WIDTH}}  {match.requirement_id}  [{kind.label}]")
        typer.echo(f"{_INDENT}{match.rationale}")
        for evidence_id in match.evidence_ids:
            typer.echo(f"{_INDENT}→ {evidence_id}")

    _print_gaps(report, kinds)


def _print_gaps(report: FitReport, kinds: dict[str, RequirementKind]) -> None:
    if not report.gaps:
        typer.echo("\nNo unmet requirements.")
        return

    # Derived from the statuses above, never generated -- a gap list that
    # disagreed with the report it came from would be unexplainable.
    typer.echo("")
    typer.secho(f"Gaps ({len(report.gaps)}):", bold=True)
    for requirement_id in report.gaps:
        typer.echo(f"  - {requirement_id} [{kinds[requirement_id].label}]")

    # Counted off the enum, not off the printed label. Comparing display text
    # would break silently the moment the label was reworded.
    unmet = sum(1 for rid in report.gaps if kinds[rid] is RequirementKind.MUST_HAVE)
    if unmet:
        typer.echo(f"\n{unmet} must-have(s) unmet.")
