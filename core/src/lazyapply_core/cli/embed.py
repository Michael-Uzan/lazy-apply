"""`lazyapply embed` — turn evidence into vectors, cheaply.

Embedding is the first step that costs money, so the command is built to be
run blind as little as possible: `--dry-run` reports exactly what would be
sent and what it would cost, before anything is sent.
"""

from __future__ import annotations

import typer

from lazyapply_core.cli._shared import (
    CacheDir,
    DataDir,
    Fake,
    NoTelemetry,
    abort,
    run_for,
    settings_for,
)
from lazyapply_core.embed import build_embedder, estimate_cost
from lazyapply_core.ingest.loader import load_evidence_dir
from lazyapply_core.ingest.parser import ParseError


def embed(
    data_dir: DataDir = None,
    cache_dir: CacheDir = None,
    fake: Fake = False,
    dry_run: bool = typer.Option(False, "--dry-run", help="Report cost, embed nothing."),
    no_telemetry: NoTelemetry = False,
) -> None:
    """Embed every evidence block, skipping anything already cached."""
    settings = settings_for(data_dir, cache_dir)
    run = run_for("embed", settings, no_telemetry=no_telemetry)

    try:
        evidence = load_evidence_dir(settings.evidence_dir)
    except ParseError as exc:
        raise abort(str(exc)) from exc

    try:
        embedder = build_embedder(settings, fake=fake, cache_dir=cache_dir, run=run)
    except ValueError as exc:
        raise abort(str(exc)) from exc

    texts = [ev.composed_text for ev in evidence]
    missing = embedder.missing(texts)

    typer.echo(f"model       {embedder.model} ({embedder.dimensions}d)")
    typer.echo(f"evidence    {len(texts)} blocks")
    typer.echo(f"cached      {len(texts) - len(missing)} cached")
    typer.echo(f"to embed    {len(missing)} to embed")
    cost = estimate_cost(missing, model=embedder.model)
    typer.echo(f"estimated cost  ${cost:.6f}")

    if dry_run:
        typer.echo("\ndry run: nothing embedded")
        return

    if not missing:
        typer.echo("\nnothing to do")
        return

    embedder.embed_documents(texts)
    typer.echo(f"\nembedded {len(missing)}, cached {len(texts)} total")
