"""An on-disk cache in front of any Embedder.

Wraps an Embedder and presents the same interface, so callers never know it is
there. Only texts with no stored vector reach the provider.

Scope: documents only. Queries are one-off, and are embedded with a different
input type hint — serving a cached query vector for a document would quietly
degrade retrieval in a way no error would reveal.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from lazyapply_core.embed.base import Embedder, Vector
from lazyapply_core.hashing import content_hash


class CachedEmbedder:
    """Embedder decorator that stores document vectors on disk."""

    def __init__(self, inner: Embedder, *, cache_dir: Path) -> None:
        self._inner = inner
        # Model and dimensions are part of the path, not just the filename.
        # Two models can return equal-length vectors that mean different
        # things, and serving one for the other produces silently wrong
        # neighbours rather than an error.
        self._dir = cache_dir / f"{inner.model}-{inner.dimensions}"

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        missing = self.missing(texts)
        if missing:
            fresh = self._inner.embed_documents(missing)
            for text, vector in zip(missing, fresh, strict=True):
                self._write(text, vector)

        # Read every vector back through one path, so a cache hit and a fresh
        # embedding cannot diverge in representation.
        return tuple(self._read(text) for text in texts)

    def embed_query(self, text: str) -> Vector:
        return self._inner.embed_query(text)

    def embed_queries(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        return self._inner.embed_queries(texts)

    def missing(self, texts: Sequence[str]) -> list[str]:
        """Texts with no stored vector, deduplicated, in first-seen order.

        Costs nothing and calls nothing — this is what `embed --dry-run`
        reports before any money is spent.
        """
        seen: dict[str, None] = {}
        for text in texts:
            if not self._path(text).exists():
                seen.setdefault(text, None)
        return list(seen)

    def _path(self, text: str) -> Path:
        return self._dir / f"{content_hash(text)}.json"

    def _write(self, text: str, vector: Vector) -> None:
        path = self._path(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file and rename: an interrupted ingest leaves
        # no half-written JSON that would fail to parse on the next run.
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(vector))
        temp.replace(path)

    def _read(self, text: str) -> Vector:
        return tuple(json.loads(self._path(text).read_text()))
