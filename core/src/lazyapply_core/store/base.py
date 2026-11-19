"""The vector store seam.

Retrieval depends on this Protocol, never on Qdrant. One contract test suite
runs against every implementation, so a test written against the in-memory
store says something true about production.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from lazyapply_core.domain import Evidence, ScoredEvidence
from lazyapply_core.embed.base import Vector


class VectorStore(Protocol):
    def ensure_collection(self, *, dimensions: int) -> None:
        """Create the collection and alias if absent. Must be idempotent —
        it runs at the start of every ingest."""

    def upsert(self, evidence: Sequence[Evidence], vectors: Sequence[Vector]) -> None:
        """Store evidence with its vectors, replacing any existing entry with
        the same evidence id. Re-running ingest must not duplicate."""

    def search(self, vector: Vector, *, limit: int) -> tuple[ScoredEvidence, ...]:
        """Nearest neighbours, highest score first, with full payloads."""

    def count(self) -> int:
        """Number of stored points."""

    def drop(self) -> None:
        """Delete the collection. Tests and reindexing only."""
