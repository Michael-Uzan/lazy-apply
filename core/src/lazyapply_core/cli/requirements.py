"""`lazyapply requirements` — see how a posting was read.

Extraction decides what gets searched for and what counts as required, so
being able to inspect it is the difference between tuning the prompt and
guessing at it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

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
from lazyapply_core.domain import RequirementKind, Role
from lazyapply_core.retrieve import ExtractionError, build_extractor, extract_role

RoleFile = Annotated[Path, typer.Argument(help="Plain-text job posting.")]


def requirements(
    role_file: RoleFile,
    data_dir: DataDir = None,
    cache_dir: CacheDir = None,
    fake: Fake = False,
    no_telemetry: NoTelemetry = False,
) -> None:
    """Extract and print the requirements for a job posting."""
    settings = settings_for(data_dir, cache_dir)
    run = run_for("requirements", settings, no_telemetry=no_telemetry)

    if not role_file.is_file():
        raise abort(f"role file not found: {role_file}")

    try:
        role = extract_role(
            role_file.read_text(),
            extractor=build_extractor(settings, fake=fake, run=run),
            cache_dir=settings.roles_cache,
        )
    except ExtractionError as exc:
        raise abort(str(exc)) from exc

    _print(role)


def _print(role: Role) -> None:
    header = role.title if not role.company else f"{role.title} — {role.company}"
    typer.echo(header)
    typer.echo(f"role_id   {role.role_id}\n")

    groups = (
        (RequirementKind.MUST_HAVE, role.must_haves),
        (RequirementKind.NICE_TO_HAVE, role.nice_to_haves),
    )
    for kind, group in groups:
        if not group:
            continue
        typer.echo(f"{kind.label} ({len(group)})")
        width = max(len(r.id) for r in group)
        for item in group:
            typer.echo(f"  {item.id:<{width}}  [{item.category.value}]  {item.text}")
        typer.echo("")
