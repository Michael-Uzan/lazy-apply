"""`lazyapply profile` — inspect the hard facts.

Everything printed here is injected verbatim into prompts. Seeing it exactly
as the model will is the point of the command.
"""

from __future__ import annotations

import typer

from lazyapply_core.cli._shared import DataDir, abort, settings_for
from lazyapply_core.domain import Profile
from lazyapply_core.ingest.parser import ParseError
from lazyapply_core.ingest.profile import load_profile, profile_version

app = typer.Typer(help="Inspect the hard facts.", no_args_is_help=True)


@app.command("show")
def show(data_dir: DataDir = None) -> None:
    """Print the validated profile and its version."""
    settings = settings_for(data_dir)

    try:
        profile = load_profile(settings.profile_path)
    except ParseError as exc:
        raise abort(str(exc)) from exc

    _print(profile)
    # Cached fit reports are keyed on this; surfacing it makes a stale cache
    # diagnosable instead of mysterious.
    typer.echo(f"\nversion   {profile_version(profile)}")


def _print(profile: Profile) -> None:
    typer.echo(f"{profile.name} — {profile.headline}")
    typer.echo(f"{profile.location} · {profile.email}")

    for link in profile.links:
        typer.echo(f"  {link.label:<10} {link.url}")

    if profile.education:
        typer.echo("\neducation")
        for item in profile.education:
            typer.echo(f"  {item.graduated}  {item.degree}, {item.institution}")

    if profile.positions:
        typer.echo("\npositions")
        for job in profile.positions:
            typer.echo(f"  {job.start} to {job.end:<8}  {job.title}, {job.company}")

    if profile.stack:
        typer.echo(f"\nstack     {', '.join(profile.stack)}")
