"""Options and helpers shared across command groups."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from lazyapply_core.config import Settings, load_settings

DataDir = Annotated[
    Path | None,
    typer.Option(
        "--data-dir",
        help="Directory holding evidence/, profile.json and roles/. Defaults to ./data.",
        show_default=False,
    ),
]


def settings_for(data_dir: Path | None) -> Settings:
    return load_settings(data_dir)


def abort(message: str) -> typer.Exit:
    """Report a user-facing failure without a traceback.

    A malformed evidence file is a mistake to fix, not a crash to debug.
    """
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=False)
    return typer.Exit(code=1)
