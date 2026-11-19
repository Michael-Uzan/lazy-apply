"""Embedding layer: one protocol, two implementations, one cache.

Callers ask for `build_embedder(settings)` and get something satisfying
`Embedder`. Whether that is Voyage, a fake, or a local model later is a
decision made here and nowhere else.
"""

from __future__ import annotations

from pathlib import Path

from lazyapply_core.config import Settings
from lazyapply_core.embed.base import Embedder, Vector
from lazyapply_core.embed.cache import CachedEmbedder
from lazyapply_core.embed.fake import FakeEmbedder
from lazyapply_core.embed.voyage import VoyageEmbedder

#: voyage-3-lite, USD per million tokens. Used for estimates only.
COST_PER_MILLION_TOKENS = 0.02

#: Rough bytes-per-token for English prose. Good enough to answer "cents or
#: dollars?" before spending; actual usage is reported by the provider.
CHARS_PER_TOKEN = 4


def build_embedder(
    settings: Settings, *, fake: bool = False, cache_dir: Path | None = None
) -> CachedEmbedder:
    inner: Embedder = (
        FakeEmbedder(dimensions=settings.vector_size)
        if fake
        else VoyageEmbedder(
            api_key=settings.voyage_api_key,
            model=settings.embedding_model,
            dimensions=settings.vector_size,
        )
    )
    return CachedEmbedder(inner, cache_dir=cache_dir or settings.cache_dir)


def estimate_cost(texts: list[str]) -> float:
    """Estimated USD to embed ``texts``."""
    tokens = sum(len(t) for t in texts) / CHARS_PER_TOKEN
    return tokens / 1_000_000 * COST_PER_MILLION_TOKENS


__all__ = [
    "CachedEmbedder",
    "Embedder",
    "FakeEmbedder",
    "Vector",
    "VoyageEmbedder",
    "build_embedder",
    "estimate_cost",
]
