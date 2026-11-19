"""Command-line entry point.

One module per command group, assembled here. Keeping the groups in separate
modules stops this from becoming the file where everything ends up.
"""

from __future__ import annotations

import typer

from lazyapply_core.cli import embed as embed_cmd
from lazyapply_core.cli import eval as eval_cmd
from lazyapply_core.cli import evidence, profile
from lazyapply_core.cli import fit as fit_cmd
from lazyapply_core.cli import ingest as ingest_cmd
from lazyapply_core.cli import pack as pack_cmd
from lazyapply_core.cli import requirements as requirements_cmd

app = typer.Typer(
    help="Lazy Apply — grounded RAG fit engine.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(evidence.app, name="evidence")
app.add_typer(profile.app, name="profile")

app.command("embed")(embed_cmd.embed)
app.command("ingest")(ingest_cmd.ingest)
app.command("search")(ingest_cmd.search)
app.command("requirements")(requirements_cmd.requirements)
app.command("pack")(pack_cmd.pack)
app.command("fit")(fit_cmd.fit)
app.command("eval")(eval_cmd.eval_)

__all__ = ["app"]
