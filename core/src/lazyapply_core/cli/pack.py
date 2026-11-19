"""`lazyapply pack` — see exactly what the matching step will be given.

The last stop before an LLM makes a judgement. If a requirement is matched
against nothing useful, the cause is visible here rather than inferred from a
confusing fit report later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from lazyapply_core.cli._shared import (
    CacheDir,
    DataDir,
    Fake,
    InMemory,
    NoTelemetry,
    abort,
    run_for,
    settings_for,
)
from lazyapply_core.domain import EvidencePack
from lazyapply_core.embed import build_embedder
from lazyapply_core.embed.base import EmbeddingError
from lazyapply_core.ingest.parser import ParseError
from lazyapply_core.ingest.pipeline import ingest as run_ingest
from lazyapply_core.retrieve import ExtractionError, build_extractor, extract_role
from lazyapply_core.retrieve.retriever import build_pack
from lazyapply_core.store import build_store

RoleFile = Annotated[Path, typer.Argument(help="Plain-text job posting.")]
TopK = Annotated[int, typer.Option("--top-k", help="Candidates retrieved per requirement.")]


def pack(
    role_file: RoleFile,
    top_k: TopK = 3,
    data_dir: DataDir = None,
    cache_dir: CacheDir = None,
    fake: Fake = False,
    in_memory: InMemory = False,
    no_telemetry: NoTelemetry = False,
) -> None:
    """Retrieve candidate evidence for every requirement in a posting."""
    settings = settings_for(data_dir, cache_dir)
    run = run_for("pack", settings, no_telemetry=no_telemetry)

    if not role_file.is_file():
        raise abort(f"role file not found: {role_file}")

    try:
        embedder = build_embedder(settings, fake=fake, run=run)
        store = build_store(settings, in_memory=in_memory)
        if in_memory and store.count() == 0:
            run_ingest(settings, embedder=embedder, store=store)

        role = extract_role(
            role_file.read_text(),
            extractor=build_extractor(settings, fake=fake, run=run),
            cache_dir=settings.roles_cache,
        )
        evidence_pack = build_pack(role, embedder=embedder, store=store, top_k=top_k)
    except (ParseError, ExtractionError, EmbeddingError, ValueError) as exc:
        raise abort(str(exc)) from exc

    typer.echo(f"{role.title}\nrole_id   {role.role_id}\n")
    _print(evidence_pack)


def _print(evidence_pack: EvidencePack) -> None:
    for item in evidence_pack.items:
        requirement = item.requirement
        typer.echo(f"{requirement.id}  [{requirement.kind.label}]  {requirement.text}")

        if not item.hits:
            typer.echo("    (nothing retrieved)")
        for hit in item.hits:
            typer.echo(f"    {hit.score:.3f}  {hit.evidence.evidence_id}")
        typer.echo("")

    unique = evidence_pack.unique_evidence()
    typer.echo(f"{len(evidence_pack.items)} requirements, {len(unique)} unique blocks to send")
