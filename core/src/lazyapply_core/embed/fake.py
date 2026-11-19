"""A deterministic embedder for tests.

Carries no semantics: "cat" and "kitten" land nowhere near each other. It
exists to exercise plumbing — caching, batching, upserts, retrieval wiring —
without a key, a network, or a cent. Retrieval quality is measured against
real embeddings in slice 9.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

from lazyapply_core.embed.base import Vector


class FakeEmbedder:
    """Maps text to a stable pseudo-random unit vector."""

    def __init__(self, *, dimensions: int = 512, model: str = "fake") -> None:
        self._dimensions = dimensions
        self._model = model
        self.calls = 0
        self.requests = 0
        """Round trips made. Batching is about this, not about `calls`."""
        """Texts embedded so far. Tests assert on it to prove the cache
        actually prevented work, rather than merely appearing to."""

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        self.requests += 1
        self.calls += len(texts)
        return tuple(self._vector(text) for text in texts)

    def embed_query(self, text: str) -> Vector:
        self.requests += 1
        self.calls += 1
        return self._vector(text)

    def embed_queries(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        self.requests += 1
        self.calls += len(texts)
        return tuple(self._vector(text) for text in texts)

    def _vector(self, text: str) -> Vector:
        # Seed from the text so the mapping is stable across processes and
        # machines; Python's own hash() is salted per process and would not be.
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        rng = random.Random(seed)
        raw = [rng.gauss(0.0, 1.0) for _ in range(self._dimensions)]
        norm = sum(v * v for v in raw) ** 0.5
        return tuple(v / norm for v in raw)
