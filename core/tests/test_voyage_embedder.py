"""The Voyage client's own logic: batching, input types, ordering.

The SDK client is injected, so these tests exercise our code with no network
and no key. The live check is a separate, explicitly-marked test.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from lazyapply_core.embed.base import EmbeddingError
from lazyapply_core.embed.voyage import VoyageEmbedder


@dataclass
class StubResponse:
    embeddings: list[list[float]]


@dataclass
class StubClient:
    """Records how it was called and returns vectors of the right shape."""

    dimensions: int = 8
    calls: list[dict[str, Any]] = field(default_factory=list)

    def embed(self, texts: list[str], *, model: str, input_type: str) -> StubResponse:
        self.calls.append({"texts": list(texts), "model": model, "input_type": input_type})
        return StubResponse([[float(i)] * self.dimensions for i, _ in enumerate(texts)])


@pytest.fixture
def stub() -> StubClient:
    return StubClient()


def embedder(stub: StubClient, **kwargs: Any) -> VoyageEmbedder:
    return VoyageEmbedder(client=stub, dimensions=8, **kwargs)


class TestInputTypes:
    def test_documents_are_embedded_as_documents(self, stub: StubClient) -> None:
        embedder(stub).embed_documents(["a"])
        assert stub.calls[0]["input_type"] == "document"

    def test_queries_are_embedded_as_queries(self, stub: StubClient) -> None:
        """Asymmetric embedding: the provider is told which side it is seeing,
        which measurably improves retrieval."""
        embedder(stub).embed_query("a")
        assert stub.calls[0]["input_type"] == "query"


class TestBatching:
    def test_short_input_is_one_request(self, stub: StubClient) -> None:
        embedder(stub, batch_size=4).embed_documents(["a", "b", "c"])
        assert len(stub.calls) == 1

    def test_long_input_is_split(self, stub: StubClient) -> None:
        embedder(stub, batch_size=2).embed_documents(["a", "b", "c", "d", "e"])
        assert [len(c["texts"]) for c in stub.calls] == [2, 2, 1]

    def test_every_text_is_sent_exactly_once(self, stub: StubClient) -> None:
        texts = [f"t{i}" for i in range(7)]
        embedder(stub, batch_size=3).embed_documents(texts)
        sent = [t for call in stub.calls for t in call["texts"]]
        assert sent == texts

    def test_batched_results_are_concatenated_in_order(self, stub: StubClient) -> None:
        vectors = embedder(stub, batch_size=2).embed_documents(["a", "b", "c", "d", "e"])
        # The stub numbers vectors by position within each batch: 0,1 | 0,1 | 0
        assert [v[0] for v in vectors] == [0.0, 1.0, 0.0, 1.0, 0.0]


class TestEdges:
    def test_empty_input_makes_no_request(self, stub: StubClient) -> None:
        assert embedder(stub).embed_documents([]) == ()
        assert stub.calls == []

    def test_returns_one_vector_per_text(self, stub: StubClient) -> None:
        assert len(embedder(stub, batch_size=2).embed_documents(["a", "b", "c"])) == 3

    def test_rejects_a_dimension_mismatch(self, stub: StubClient) -> None:
        """A provider returning an unexpected width would poison the whole
        collection; fail on the first vector instead."""
        with pytest.raises(ValueError, match="dimensions"):
            VoyageEmbedder(client=stub, dimensions=512).embed_documents(["a"])

    def test_requires_an_api_key_when_building_its_own_client(self) -> None:
        with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
            VoyageEmbedder(api_key="")


@dataclass
class FailingClient:
    """Stands in for any provider-side failure: rate limit, auth, outage."""

    error: Exception

    def embed(self, texts: list[str], *, model: str, input_type: str) -> StubResponse:
        raise self.error


class TestProviderFailures:
    """Provider errors are expected operating conditions, not crashes. They
    must surface as EmbeddingError so the CLI can print one line."""

    def test_rate_limit_becomes_an_embedding_error(self) -> None:
        client = FailingClient(RuntimeError("rate limit: 3 RPM"))
        with pytest.raises(EmbeddingError, match="rate limit"):
            VoyageEmbedder(client=client, dimensions=8).embed_documents(["a"])

    def test_query_failures_too(self) -> None:
        client = FailingClient(RuntimeError("boom"))
        with pytest.raises(EmbeddingError):
            VoyageEmbedder(client=client, dimensions=8).embed_query("a")

    def test_the_original_error_is_kept_as_the_cause(self) -> None:
        original = RuntimeError("underlying detail")
        client = FailingClient(original)
        with pytest.raises(EmbeddingError) as info:
            VoyageEmbedder(client=client, dimensions=8).embed_documents(["a"])
        assert info.value.__cause__ is original

    def test_a_dimension_mismatch_is_not_swallowed_as_a_provider_error(
        self, stub: StubClient
    ) -> None:
        """Our own validation must stay distinguishable from provider faults."""
        with pytest.raises(ValueError, match="dimensions"):
            VoyageEmbedder(client=stub, dimensions=512).embed_documents(["a"])
