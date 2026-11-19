"""The ingest pipeline.

    evidence/*.md -> parse -> compose -> embed (cache-aware) -> upsert

Idempotent end to end: unchanged evidence costs no embedding call and creates
no duplicate point. Running it is always safe.
"""

from __future__ import annotations

from dataclasses import dataclass

from lazyapply_core.config import Settings
from lazyapply_core.embed.cache import CachedEmbedder
from lazyapply_core.ingest.loader import load_evidence_dir
from lazyapply_core.store.base import VectorStore


@dataclass(frozen=True)
class IngestResult:
    blocks: int
    """Evidence blocks parsed."""

    embedded: int
    """Blocks sent to the provider. Zero on an unchanged re-run."""

    cached: int
    """Blocks served from the embedding cache."""


def ingest(settings: Settings, *, embedder: CachedEmbedder, store: VectorStore) -> IngestResult:
    evidence = load_evidence_dir(settings.evidence_dir)
    texts = [ev.composed_text for ev in evidence]

    # Counted before embedding, since the cache fills as a side effect of it.
    missing = embedder.missing(texts)

    store.ensure_collection(dimensions=embedder.dimensions)
    store.upsert(evidence, embedder.embed_documents(texts))

    return IngestResult(
        blocks=len(evidence),
        embedded=len(missing),
        cached=len(texts) - len(missing),
    )
