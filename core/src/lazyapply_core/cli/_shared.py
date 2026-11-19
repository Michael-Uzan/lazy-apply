"""Options and helpers shared across command groups."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from lazyapply_core.config import Settings, load_settings
from lazyapply_core.telemetry import Run

DataDir = Annotated[
    Path | None,
    typer.Option(
        "--data-dir",
        help="Directory holding evidence/, profile.json and roles/. Defaults to ./data.",
        show_default=False,
    ),
]


CacheDir = Annotated[
    Path | None,
    typer.Option(
        "--cache-dir",
        help="Where embedding vectors are cached. Defaults to ./.cache.",
        show_default=False,
    ),
]

Fake = Annotated[
    bool,
    typer.Option("--fake", help="Use the fake embedder: no API key, no network, no cost."),
]


InMemory = Annotated[
    bool,
    typer.Option("--in-memory", help="Use an in-process vector store instead of Qdrant."),
]


NoTelemetry = Annotated[
    bool,
    typer.Option("--no-telemetry", help="Do not record this run in the usage log."),
]


def run_for(command: str, settings: Settings, *, no_telemetry: bool = False) -> Run:
    """Telemetry context for one command invocation.

    One id per invocation, shared by every provider call it makes -- that is
    what lets `lazyapply cost` report what a command costs rather than what a
    single API call costs.
    """
    return Run.for_command(command, settings, enabled=not no_telemetry)


def settings_for(data_dir: Path | None, cache_dir: Path | None = None) -> Settings:
    """Resolve both path overrides onto settings, once per command.

    After this, ``settings.cache_dir`` is authoritative -- nothing downstream
    takes a separate cache path.
    """
    return load_settings(data_dir, cache_dir)


def abort(message: str) -> typer.Exit:
    """Report a user-facing failure without a traceback.

    A malformed evidence file is a mistake to fix, not a crash to debug.
    """
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=False)
    return typer.Exit(code=1)
