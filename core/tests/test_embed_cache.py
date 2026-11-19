"""On-disk embedding cache.

Ingest runs hundreds of times while building. Re-embedding unchanged evidence
every run is pure waste, so vectors are cached by content hash and only misses
reach the provider.
"""

from pathlib import Path

import pytest

from lazyapply_core.embed.cache import CachedEmbedder
from lazyapply_core.embed.fake import FakeEmbedder


@pytest.fixture
def inner() -> FakeEmbedder:
    return FakeEmbedder(dimensions=8)


class TestCaching:
    def test_first_call_reaches_the_provider(self, inner: FakeEmbedder, tmp_path: Path) -> None:
        CachedEmbedder(inner, cache_dir=tmp_path).embed_documents(["a", "b"])
        assert inner.calls == 2

    def test_second_call_does_not(self, inner: FakeEmbedder, tmp_path: Path) -> None:
        cached = CachedEmbedder(inner, cache_dir=tmp_path)
        cached.embed_documents(["a", "b"])
        cached.embed_documents(["a", "b"])
        assert inner.calls == 2

    def test_only_misses_reach_the_provider(self, inner: FakeEmbedder, tmp_path: Path) -> None:
        cached = CachedEmbedder(inner, cache_dir=tmp_path)
        cached.embed_documents(["a"])
        cached.embed_documents(["a", "b"])
        assert inner.calls == 2

    def test_cached_vectors_equal_the_originals(self, inner: FakeEmbedder, tmp_path: Path) -> None:
        cached = CachedEmbedder(inner, cache_dir=tmp_path)
        first = cached.embed_documents(["a"])
        second = cached.embed_documents(["a"])
        assert first == second

    def test_results_stay_in_input_order(self, inner: FakeEmbedder, tmp_path: Path) -> None:
        """A partial cache hit must not reorder results."""
        cached = CachedEmbedder(inner, cache_dir=tmp_path)
        cached.embed_documents(["b"])
        direct = FakeEmbedder(dimensions=8).embed_documents(["a", "b", "c"])
        assert cached.embed_documents(["a", "b", "c"]) == direct

    def test_survives_a_new_process(self, tmp_path: Path) -> None:
        CachedEmbedder(FakeEmbedder(dimensions=8), cache_dir=tmp_path).embed_documents(["a"])
        second = FakeEmbedder(dimensions=8)
        CachedEmbedder(second, cache_dir=tmp_path).embed_documents(["a"])
        assert second.calls == 0


class TestCacheKeying:
    def test_different_models_do_not_share_entries(self, tmp_path: Path) -> None:
        """Vectors from two models are not interchangeable. Sharing a cache
        across them would silently return the wrong vector."""
        CachedEmbedder(FakeEmbedder(model="m1", dimensions=8), cache_dir=tmp_path).embed_documents(
            ["a"]
        )
        other = FakeEmbedder(model="m2", dimensions=8)
        CachedEmbedder(other, cache_dir=tmp_path).embed_documents(["a"])
        assert other.calls == 1

    def test_queries_are_not_cached(self, inner: FakeEmbedder, tmp_path: Path) -> None:
        """Queries are one-off and embedded with a different input type; a
        query vector must never be served for a document."""
        cached = CachedEmbedder(inner, cache_dir=tmp_path)
        cached.embed_query("a")
        cached.embed_query("a")
        assert inner.calls == 2


class TestReporting:
    def test_missing_reports_uncached_texts_without_embedding(
        self, inner: FakeEmbedder, tmp_path: Path
    ) -> None:
        """Powers `embed --dry-run`: what would this cost, before spending it."""
        cached = CachedEmbedder(inner, cache_dir=tmp_path)
        cached.embed_documents(["a"])
        assert cached.missing(["a", "b", "c"]) == ["b", "c"]
        assert inner.calls == 1

    def test_missing_deduplicates(self, inner: FakeEmbedder, tmp_path: Path) -> None:
        cached = CachedEmbedder(inner, cache_dir=tmp_path)
        assert cached.missing(["b", "b"]) == ["b"]

    def test_duplicate_texts_are_embedded_once(self, inner: FakeEmbedder, tmp_path: Path) -> None:
        cached = CachedEmbedder(inner, cache_dir=tmp_path)
        cached.embed_documents(["a", "a", "a"])
        assert inner.calls == 1
