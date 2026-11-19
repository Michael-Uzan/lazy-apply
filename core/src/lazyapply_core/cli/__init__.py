"""Command-line entry point.

One module per command group, assembled here. Keeping the groups in separate
modules stops this from becoming the file where everything ends up.
"""

from __future__ import annotations

import typer

from lazyapply_core.cli import embed as embed_cmd
from lazyapply_core.cli import evidence, profile
from lazyapply_core.cli import ingest as ingest_cmd

app = typer.Typer(
    help="Lazy Apply — grounded RAG fit engine.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(evidence.app, name="evidence")
app.command("embed")(embed_cmd.embed)
app.command("ingest")(ingest_cmd.ingest)
app.command("search")(ingest_cmd.search)
app.add_typer(profile.app, name="profile")

__all__ = ["app"]
