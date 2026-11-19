"""In-memory vector store.

Exists so the ingest and retrieval pipelines can be tested without a running
Qdrant. It satisfies the same contract test as QdrantStore — brute-force
cosine instead of HNSW, which is exact and perfectly adequate at test sizes.
"""

from __future__ import annotations

from collections.abc import Sequence

from lazyapply_core.domain import Evidence, ScoredEvidence
from lazyapply_core.embed.base import Vector


class InMemoryStore:
    def __init__(self, *, collection: str, alias: str) -> None:
        self.collection = collection
        self.alias = alias
        self._points: dict[str, tuple[Evidence, Vector]] = {}
        self._dimensions: int | None = None

    def ensure_collection(self, *, dimensions: int) -> None:
        self._dimensions = dimensions

    def upsert(self, evidence: Sequence[Evidence], vectors: Sequence[Vector]) -> None:
        for item, vector in zip(evidence, vectors, strict=True):
            # Keyed by evidence id, so a re-upsert replaces rather than adds.
            self._points[item.evidence_id] = (item, vector)

    def search(self, vector: Vector, *, limit: int) -> tuple[ScoredEvidence, ...]:
        scored = [
            ScoredEvidence(evidence=item, score=_cosine(vector, stored))
            for item, stored in self._points.values()
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return tuple(scored[:limit])

    def count(self) -> int:
        return len(self._points)

    def drop(self) -> None:
        self._points.clear()


def _cosine(a: Vector, b: Vector) -> float:
    """Dot product. Both sides are unit vectors, so this is cosine similarity."""
    return sum(x * y for x, y in zip(a, b, strict=True))
