"""The fake embedder: deterministic stand-in so tests need no API key.

It carries no semantics — similar sentences do not get similar vectors. It
exists to exercise plumbing (caching, batching, upserts), never retrieval
quality. Quality is measured in slice 9 against real embeddings.
"""

from lazyapply_core.embed.fake import FakeEmbedder


class TestDeterminism:
    def test_same_text_gives_the_same_vector(self) -> None:
        a, b = FakeEmbedder(), FakeEmbedder()
        assert a.embed_documents(["hello"]) == b.embed_documents(["hello"])

    def test_different_text_gives_a_different_vector(self) -> None:
        e = FakeEmbedder()
        assert e.embed_documents(["hello"]) != e.embed_documents(["goodbye"])


class TestShape:
    def test_returns_one_vector_per_text(self) -> None:
        assert len(FakeEmbedder().embed_documents(["a", "b", "c"])) == 3

    def test_vectors_have_the_configured_dimensions(self) -> None:
        e = FakeEmbedder(dimensions=512)
        assert len(e.embed_documents(["x"])[0]) == 512

    def test_vectors_are_unit_length(self) -> None:
        """Cosine distance assumes normalized vectors; the fake matches that."""
        (vector,) = FakeEmbedder().embed_documents(["x"])
        assert abs(sum(v * v for v in vector) ** 0.5 - 1.0) < 1e-9

    def test_empty_input_returns_empty_output(self) -> None:
        assert FakeEmbedder().embed_documents([]) == ()


class TestCallCounting:
    def test_counts_texts_embedded(self) -> None:
        """Tests assert on this to prove the cache avoided real work."""
        e = FakeEmbedder()
        e.embed_documents(["a", "b"])
        e.embed_query("c")
        assert e.calls == 3
