"""Command-line entry point.

One module per command group, assembled here. Keeping the groups in separate
modules stops this from becoming the file where everything ends up.
"""

from __future__ import annotations

import typer

from lazyapply_core.cli import evidence

app = typer.Typer(
    help="Lazy Apply — grounded RAG fit engine.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(evidence.app, name="evidence")

__all__ = ["app"]
