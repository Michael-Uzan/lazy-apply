"""Voyage AI embeddings.

The only module in the project that knows Voyage exists. Everything else
depends on the Embedder protocol, so replacing this with a local model or a
different provider touches one file and one line of wiring.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from time import perf_counter
from typing import Any

from lazyapply_core.embed.base import EmbeddingError, Vector
from lazyapply_core.telemetry.base import NullTelemetry, Telemetry
from lazyapply_core.telemetry.providers import record_voyage

DEFAULT_MODEL = "voyage-3-lite"
DEFAULT_DIMENSIONS = 512

#: Voyage accepts up to 128 texts per request.
DEFAULT_BATCH_SIZE = 128

DOCUMENT = "document"
"""Embeds text as something to be found."""

QUERY = "query"
"""Embeds text as something doing the finding.

Voyage is an asymmetric model: the same sentence embeds differently depending
on which side of the search it is on. A terse requirement ("Built production
React applications") and a paragraph-long impact story are not the same shape
of text, and telling the model which is which is free accuracy. Getting this
backwards costs retrieval quality silently -- nothing errors."""


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
        telemetry: Telemetry | None = None,
        command: str = "embed",
        run: str = "",
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
        self._telemetry = telemetry or NullTelemetry()
        self._command = command
        self._run = run

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        vectors: list[Vector] = []
        for batch in _batched(texts, self._batch_size):
            vectors.extend(self._embed(batch, input_type=DOCUMENT))
        return tuple(vectors)

    def embed_query(self, text: str) -> Vector:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        vectors: list[Vector] = []
        for batch in _batched(texts, self._batch_size):
            vectors.extend(self._embed(batch, input_type=QUERY))
        return tuple(vectors)

    def _embed(self, texts: Sequence[str], *, input_type: str) -> tuple[Vector, ...]:
        if not texts:
            return ()

        started = perf_counter()
        try:
            response = self._client.embed(list(texts), model=self._model, input_type=input_type)
        except Exception as exc:
            raise EmbeddingError(f"{self._model}: {exc}") from exc

        record_voyage(
            self._telemetry,
            response,
            command=self._command,
            run=self._run,
            model=self._model,
            started=started,
        )

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
