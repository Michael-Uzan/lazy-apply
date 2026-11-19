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
from lazyapply_core.embed.tokens import estimate_tokens
from lazyapply_core.embed.voyage import VoyageEmbedder
from lazyapply_core.telemetry import Run
from lazyapply_core.telemetry.pricing import cost_usd


def build_embedder(
    settings: Settings,
    *,
    fake: bool = False,
    cache_dir: Path | None = None,
    run: Run | None = None,
) -> CachedEmbedder:
    context = run or Run.off()
    inner: Embedder = (
        FakeEmbedder(dimensions=settings.vector_size)
        if fake
        else VoyageEmbedder(
            api_key=settings.voyage_api_key,
            model=settings.embedding_model,
            dimensions=settings.vector_size,
            telemetry=context.telemetry,
            command=context.command,
            run=context.id,
        )
    )
    return CachedEmbedder(
        inner,
        cache_dir=cache_dir or settings.cache_dir,
        telemetry=context.telemetry,
        command=context.command,
        run=context.id,
    )


def estimate_cost(texts: list[str], *, model: str) -> float:
    """Estimated USD to embed ``texts``.

    Priced from the same table the cost report uses, so an estimate and a
    bill can never disagree about what a model charges.
    """
    return cost_usd(model, input_tokens=estimate_tokens(texts), output_tokens=0)


__all__ = [
    "CachedEmbedder",
    "Embedder",
    "FakeEmbedder",
    "Vector",
    "VoyageEmbedder",
    "build_embedder",
    "estimate_cost",
]
