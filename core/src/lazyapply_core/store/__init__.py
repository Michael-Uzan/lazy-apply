"""Vector store layer: one protocol, two implementations.

`build_store` is the only place that decides which one exists.
"""

from __future__ import annotations

from lazyapply_core.config import Settings
from lazyapply_core.store.base import VectorStore
from lazyapply_core.store.memory import InMemoryStore

#: One in-memory store per process, so `ingest` and `search` in the same CLI
#: invocation see the same data. Used by tests and `--in-memory`.
_SHARED_MEMORY_STORE: InMemoryStore | None = None


def build_store(settings: Settings, *, in_memory: bool = False) -> VectorStore:
    if in_memory:
        global _SHARED_MEMORY_STORE
        if _SHARED_MEMORY_STORE is None:
            _SHARED_MEMORY_STORE = InMemoryStore(
                collection=settings.collection_version, alias=settings.collection_alias
            )
        return _SHARED_MEMORY_STORE

    from qdrant_client import QdrantClient

    from lazyapply_core.store.qdrant import QdrantStore

    return QdrantStore(
        client=QdrantClient(url=settings.qdrant_url),
        collection=settings.collection_version,
        alias=settings.collection_alias,
    )


__all__ = ["InMemoryStore", "VectorStore", "build_store"]
