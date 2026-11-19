"""Voyage AI embeddings.

The only module in the project that knows Voyage exists. Everything else
depends on the Embedder protocol, so replacing this with a local model or a
different provider touches one file and one line of wiring.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from lazyapply_core.embed.base import EmbeddingError, Vector

DEFAULT_MODEL = "voyage-3-lite"
DEFAULT_DIMENSIONS = 512

#: Voyage accepts up to 128 texts per request.
DEFAULT_BATCH_SIZE = 128


class VoyageEmbedder:
    """Embedder backed by the Voyage API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        client: Any | None = None,
    ) -> None:
        """
        Args:
            client: Pre-built SDK client. Injected by tests so this class's own
                logic — batching, input types, ordering — is exercised without
                a network or a key.
        """
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._client = client if client is not None else _build_client(api_key)

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        vectors: list[Vector] = []
        for batch in _batched(texts, self._batch_size):
            vectors.extend(self._embed(batch, input_type="document"))
        return tuple(vectors)

    def embed_query(self, text: str) -> Vector:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        vectors: list[Vector] = []
        for batch in _batched(texts, self._batch_size):
            vectors.extend(self._embed(batch, input_type="query"))
        return tuple(vectors)

    def _embed(self, texts: Sequence[str], *, input_type: str) -> tuple[Vector, ...]:
        if not texts:
            return ()

        try:
            response = self._client.embed(list(texts), model=self._model, input_type=input_type)
        except Exception as exc:
            raise EmbeddingError(f"{self._model}: {exc}") from exc

        vectors = tuple(tuple(float(v) for v in e) for e in response.embeddings)

        for vector in vectors:
            if len(vector) != self._dimensions:
                raise ValueError(
                    f"{self._model} returned {len(vector)} dimensions, expected {self._dimensions}"
                )
        return vectors


def _batched(texts: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(texts), size):
        yield texts[start : start + size]


def _build_client(api_key: str | None) -> Any:
    if not api_key:
        raise ValueError("VOYAGE_API_KEY is not set; add it to .env")

    import voyageai

    # The SDK has no explicit __all__, so mypy cannot see Client.
    return voyageai.Client(api_key=api_key)  # type: ignore[attr-defined]
