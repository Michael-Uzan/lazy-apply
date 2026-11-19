"""The embedding seam.

Everything downstream depends on this Protocol, never on Voyage. That is what
lets the whole pipeline run in tests with no API key, and what makes swapping
in a local model later a contained change.

Documents and queries are embedded through separate methods on purpose.
Providers accept an input type hint, and using it measurably improves
retrieval: a stored impact story and a terse requirement query are asymmetric,
and the model is told which side it is looking at.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

Vector = tuple[float, ...]


class EmbeddingError(Exception):
    """The embedding provider failed.

    Rate limits, auth failures and outages are normal operating conditions
    for a networked dependency, not bugs. They get their own type so the
    CLI can report one line instead of a traceback, while a genuine defect
    in our own code still raises loudly.
    """


@runtime_checkable
class Embedder(Protocol):
    @property
    def model(self) -> str:
        """Provider's model id. Part of every cache key — vectors from two
        models are not interchangeable, even at equal dimensions."""

    @property
    def dimensions(self) -> int:
        """Vector length. Must match the Qdrant collection's configured size."""

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        """Embed stored evidence. Returns one vector per input, in order."""

    def embed_query(self, text: str) -> Vector:
        """Embed a single search query."""

    def embed_queries(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        """Embed several queries in one round trip.

        A role has roughly ten requirements and each is a query. One
        request each would be ten round trips, which a rate-limited free
        tier turns into minutes of waiting.
        """
