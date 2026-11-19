"""Batched query embedding.

A role has ~10 requirements and each one is a search query. One request per
requirement would be ten round trips -- over three minutes on a 3 RPM free
tier -- so they go in a single batched call.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from lazyapply_core.embed.fake import FakeEmbedder
from lazyapply_core.embed.voyage import VoyageEmbedder


class TestFakeEmbedder:
    def test_returns_one_vector_per_query(self) -> None:
        assert len(FakeEmbedder().embed_queries(["a", "b", "c"])) == 3

    def test_matches_embed_query_for_a_single_text(self) -> None:
        e = FakeEmbedder()
        assert e.embed_queries(["a"])[0] == e.embed_query("a")

    def test_empty_input_returns_empty_output(self) -> None:
        assert FakeEmbedder().embed_queries([]) == ()

    def test_counts_one_request_regardless_of_batch_size(self) -> None:
        """`requests` counts round trips; `calls` counts texts. Batching is
        about the former."""
        e = FakeEmbedder()
        e.embed_queries(["a", "b", "c"])
        assert e.requests == 1
        assert e.calls == 3


@dataclass
class StubResponse:
    embeddings: list[list[float]]


@dataclass
class StubClient:
    dimensions: int = 8
    calls: list[dict[str, Any]] = field(default_factory=list)

    def embed(self, texts: list[str], *, model: str, input_type: str) -> StubResponse:
        self.calls.append({"texts": list(texts), "input_type": input_type})
        return StubResponse([[float(i)] * self.dimensions for i, _ in enumerate(texts)])


@pytest.fixture
def stub() -> StubClient:
    return StubClient()


class TestVoyageEmbedder:
    def test_sends_one_request_for_many_queries(self, stub: StubClient) -> None:
        VoyageEmbedder(client=stub, dimensions=8).embed_queries(["a", "b", "c"])
        assert len(stub.calls) == 1

    def test_embeds_them_as_queries_not_documents(self, stub: StubClient) -> None:
        VoyageEmbedder(client=stub, dimensions=8).embed_queries(["a", "b"])
        assert stub.calls[0]["input_type"] == "query"

    def test_respects_the_batch_size(self, stub: StubClient) -> None:
        VoyageEmbedder(client=stub, dimensions=8, batch_size=2).embed_queries(["a", "b", "c"])
        assert [len(c["texts"]) for c in stub.calls] == [2, 1]

    def test_makes_no_request_for_empty_input(self, stub: StubClient) -> None:
        assert VoyageEmbedder(client=stub, dimensions=8).embed_queries([]) == ()


class TestCachedEmbedder:
    def test_passes_queries_through_uncached(self, tmp_path) -> None:
        """Queries stay uncached -- they are one-off and use a different
        input type, so serving one for a document would degrade retrieval."""
        from lazyapply_core.embed.cache import CachedEmbedder

        inner = FakeEmbedder(dimensions=8)
        cached = CachedEmbedder(inner, cache_dir=tmp_path)
        cached.embed_queries(["a", "b"])
        cached.embed_queries(["a", "b"])
        assert inner.requests == 2
        assert list(tmp_path.rglob("*.json")) == []
