"""One suite, run against every VectorStore implementation.

If the in-memory store and Qdrant both pass this, then tests written against
the fake mean something about production. The Qdrant parameter skips itself
when no server is reachable, so the suite stays runnable offline.
"""

from collections.abc import Iterator

import pytest

from lazyapply_core.domain import Evidence
from lazyapply_core.embed.fake import FakeEmbedder
from lazyapply_core.store.base import VectorStore
from lazyapply_core.store.memory import InMemoryStore

DIMENSIONS = 8


def make_evidence(slug: str, text: str) -> Evidence:
    return Evidence(
        evidence_id=f"ev_test__{slug}",
        title=slug.replace("-", " "),
        body=text,
        composed_text=text,
        content_hash="0" * 16,
        company="Test Co",
        tags=("alpha", "beta"),
        source_file="test.md",
    )


@pytest.fixture(params=["memory", "qdrant"])
def store(request: pytest.FixtureRequest) -> Iterator[VectorStore]:
    if request.param == "memory":
        yield InMemoryStore(collection="t", alias="t")
        return

    qdrant = pytest.importorskip("qdrant_client")
    from lazyapply_core.store.qdrant import QdrantStore

    name = f"contract_test_{request.node.name[:20]}"
    try:
        client = qdrant.QdrantClient(url="http://localhost:6333", timeout=2)
        client.get_collections()
    except Exception:
        pytest.skip("no Qdrant on localhost:6333")

    s = QdrantStore(client=client, collection=f"{name}_v1", alias=name)
    try:
        yield s
    finally:
        s.drop()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dimensions=DIMENSIONS)


def seed(store: VectorStore, embedder: FakeEmbedder, items: list[tuple[str, str]]) -> None:
    store.ensure_collection(dimensions=DIMENSIONS)
    evidence = tuple(make_evidence(slug, text) for slug, text in items)
    vectors = embedder.embed_documents([e.composed_text for e in evidence])
    store.upsert(evidence, vectors)


class TestLifecycle:
    def test_starts_empty(self, store: VectorStore) -> None:
        store.ensure_collection(dimensions=DIMENSIONS)
        assert store.count() == 0

    def test_ensure_collection_is_idempotent(self, store: VectorStore) -> None:
        store.ensure_collection(dimensions=DIMENSIONS)
        store.ensure_collection(dimensions=DIMENSIONS)
        assert store.count() == 0


class TestUpsert:
    def test_counts_what_was_stored(self, store: VectorStore, embedder: FakeEmbedder) -> None:
        seed(store, embedder, [("a", "alpha text"), ("b", "beta text")])
        assert store.count() == 2

    def test_reupserting_replaces_rather_than_duplicates(
        self, store: VectorStore, embedder: FakeEmbedder
    ) -> None:
        """Re-running ingest must not grow the collection."""
        seed(store, embedder, [("a", "alpha text")])
        seed(store, embedder, [("a", "alpha text")])
        assert store.count() == 1


class TestSearch:
    def test_finds_the_matching_evidence(self, store: VectorStore, embedder: FakeEmbedder) -> None:
        seed(store, embedder, [("a", "alpha text"), ("b", "beta text")])
        hits = store.search(embedder.embed_query("alpha text"), limit=1)
        assert hits[0].evidence.evidence_id == "ev_test__a"

    def test_respects_the_limit(self, store: VectorStore, embedder: FakeEmbedder) -> None:
        seed(store, embedder, [("a", "a"), ("b", "b"), ("c", "c")])
        assert len(store.search(embedder.embed_query("a"), limit=2)) == 2

    def test_returns_scores_in_descending_order(
        self, store: VectorStore, embedder: FakeEmbedder
    ) -> None:
        seed(store, embedder, [("a", "a"), ("b", "b"), ("c", "c")])
        scores = [h.score for h in store.search(embedder.embed_query("a"), limit=3)]
        assert scores == sorted(scores, reverse=True)

    def test_round_trips_the_full_payload(self, store: VectorStore, embedder: FakeEmbedder) -> None:
        """Search must return usable Evidence, not ids needing a second lookup."""
        seed(store, embedder, [("a", "alpha text")])
        evidence = store.search(embedder.embed_query("alpha text"), limit=1)[0].evidence
        assert evidence.company == "Test Co"
        assert evidence.tags == ("alpha", "beta")
        assert evidence.body == "alpha text"
        assert evidence.source_file == "test.md"

    def test_empty_collection_returns_nothing(
        self, store: VectorStore, embedder: FakeEmbedder
    ) -> None:
        store.ensure_collection(dimensions=DIMENSIONS)
        assert store.search(embedder.embed_query("anything"), limit=3) == ()
