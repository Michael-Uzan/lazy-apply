"""`lazyapply ingest` and `lazyapply search`.

Ingest is safe to re-run: unchanged evidence costs nothing and duplicates
nothing. Search is the first command where retrieval is visible, and exists
mainly so you can eyeball whether it is sane before trusting it in slice 7.
"""

from __future__ import annotations

import typer

from lazyapply_core.cli._shared import CacheDir, DataDir, Fake, InMemory, abort, settings_for
from lazyapply_core.embed import build_embedder
from lazyapply_core.embed.base import EmbeddingError
from lazyapply_core.ingest.parser import ParseError
from lazyapply_core.ingest.pipeline import ingest as run_ingest
from lazyapply_core.store import build_store


def ingest(
    data_dir: DataDir = None,
    cache_dir: CacheDir = None,
    fake: Fake = False,
    in_memory: InMemory = False,
) -> None:
    """Parse, embed and store every evidence block."""
    settings = settings_for(data_dir)

    try:
        embedder = build_embedder(settings, fake=fake, cache_dir=cache_dir)
        store = build_store(settings, in_memory=in_memory)
        result = run_ingest(settings, embedder=embedder, store=store)
    except (ParseError, ValueError, EmbeddingError) as exc:
        raise abort(str(exc)) from exc

    typer.echo(f"parsed    {result.blocks} blocks")
    typer.echo(f"embedded  {result.embedded} embedded")
    typer.echo(f"cached    {result.cached} cached")
    typer.echo(f"stored    {store.count()} points in '{settings.collection_alias}'")


def search(
    query: str,
    limit: int = typer.Option(5, "--limit", "-n", help="How many results."),
    data_dir: DataDir = None,
    cache_dir: CacheDir = None,
    fake: Fake = False,
    in_memory: InMemory = False,
) -> None:
    """Semantic search over stored evidence."""
    settings = settings_for(data_dir)

    try:
        embedder = build_embedder(settings, fake=fake, cache_dir=cache_dir)
        store = build_store(settings, in_memory=in_memory)
        if in_memory and store.count() == 0:
            run_ingest(settings, embedder=embedder, store=store)
        hits = store.search(embedder.embed_query(query), limit=limit)
    except (ParseError, ValueError, EmbeddingError) as exc:
        raise abort(str(exc)) from exc

    if not hits:
        typer.echo("no results -- has `lazyapply ingest` been run?")
        return

    width = max(len(h.evidence.evidence_id) for h in hits)
    for hit in hits:
        typer.echo(f"{hit.score:.3f}  {hit.evidence.evidence_id:<{width}}  {hit.evidence.title}")
