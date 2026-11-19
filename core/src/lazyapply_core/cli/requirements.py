"""`lazyapply requirements` — see how a posting was read.

Extraction decides what gets searched for and what counts as required, so
being able to inspect it is the difference between tuning the prompt and
guessing at it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from lazyapply_core.cli._shared import CacheDir, DataDir, Fake, abort, settings_for
from lazyapply_core.domain import Role
from lazyapply_core.retrieve import ExtractionError, build_extractor, extract_role

RoleFile = Annotated[Path, typer.Argument(help="Plain-text job posting.")]


def requirements(
    role_file: RoleFile,
    data_dir: DataDir = None,
    cache_dir: CacheDir = None,
    fake: Fake = False,
) -> None:
    """Extract and print the requirements for a job posting."""
    settings = settings_for(data_dir)

    if not role_file.is_file():
        raise abort(f"role file not found: {role_file}")

    try:
        role = extract_role(
            role_file.read_text(),
            extractor=build_extractor(settings, fake=fake),
            cache_dir=(cache_dir or settings.cache_dir) / "roles",
        )
    except ExtractionError as exc:
        raise abort(str(exc)) from exc

    _print(role)


def _print(role: Role) -> None:
    header = role.title if not role.company else f"{role.title} — {role.company}"
    typer.echo(header)
    typer.echo(f"role_id   {role.role_id}\n")

    for label, group in (("must-have", role.must_haves), ("nice-to-have", role.nice_to_haves)):
        if not group:
            continue
        typer.echo(f"{label} ({len(group)})")
        width = max(len(r.id) for r in group)
        for item in group:
            typer.echo(f"  {item.id:<{width}}  [{item.category.value}]  {item.text}")
        typer.echo("")
