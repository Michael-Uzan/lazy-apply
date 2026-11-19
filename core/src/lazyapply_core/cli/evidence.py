"""`lazyapply evidence` — inspect what the parser made of your markdown.

These commands cost nothing to run and need no API key. They exist so you can
see exactly what will be embedded before paying to embed it.
"""

from __future__ import annotations

import typer

from lazyapply_core.cli._shared import DataDir, abort, settings_for
from lazyapply_core.domain import Evidence
from lazyapply_core.ingest.loader import load_evidence_dir
from lazyapply_core.ingest.parser import ParseError

app = typer.Typer(help="Inspect parsed evidence.", no_args_is_help=True)


def _load(data_dir: DataDir) -> tuple[Evidence, ...]:
    settings = settings_for(data_dir)
    try:
        return load_evidence_dir(settings.evidence_dir)
    except ParseError as exc:
        raise abort(str(exc)) from exc


@app.command("list")
def list_evidence(data_dir: DataDir = None) -> None:
    """List every parsed evidence block."""
    evidence = _load(data_dir)

    width = max(len(ev.evidence_id) for ev in evidence)
    for ev in evidence:
        tags = ", ".join(ev.tags) or "-"
        typer.echo(f"{ev.evidence_id:<{width}}  {len(ev.body):>5} chars  {tags}")

    files = len({ev.source_file for ev in evidence})
    typer.echo(f"\n{len(evidence)} blocks from {files} file(s)")


@app.command("show")
def show_evidence(evidence_id: str, data_dir: DataDir = None) -> None:
    """Print the exact text that will be embedded for one block."""
    evidence = _load(data_dir)

    match = next((ev for ev in evidence if ev.evidence_id == evidence_id), None)
    if match is None:
        raise abort(f"no evidence with id {evidence_id!r}")

    typer.echo(f"source      {match.source_file}")
    typer.echo(f"hash        {match.content_hash}")
    typer.echo("\n--- embedded text ---")
    typer.echo(match.composed_text)
