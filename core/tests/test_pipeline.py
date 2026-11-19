"""The ingest pipeline: markdown on disk to vectors in a store."""

from pathlib import Path

import pytest

from lazyapply_core.config import REPO_ROOT, load_settings
from lazyapply_core.embed.cache import CachedEmbedder
from lazyapply_core.embed.fake import FakeEmbedder
from lazyapply_core.ingest.pipeline import ingest
from lazyapply_core.store.memory import InMemoryStore

EXAMPLE = REPO_ROOT / "data/example"


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore(collection="evidence_v1", alias="evidence")


@pytest.fixture
def embedder(tmp_path: Path) -> CachedEmbedder:
    return CachedEmbedder(FakeEmbedder(dimensions=512), cache_dir=tmp_path)


@pytest.fixture
def settings():
    return load_settings(EXAMPLE)


class TestFirstRun:
    def test_reports_every_block(self, settings, embedder, store) -> None:
        assert ingest(settings, embedder=embedder, store=store).blocks == 6

    def test_embeds_everything(self, settings, embedder, store) -> None:
        assert ingest(settings, embedder=embedder, store=store).embedded == 6

    def test_stores_everything(self, settings, embedder, store) -> None:
        ingest(settings, embedder=embedder, store=store)
        assert store.count() == 6

    def test_creates_the_collection(self, settings, embedder, store) -> None:
        """Ingest must work against an empty store with no manual setup."""
        ingest(settings, embedder=embedder, store=store)
        assert store.count() > 0


class TestSecondRun:
    def test_embeds_nothing_new(self, settings, embedder, store) -> None:
        ingest(settings, embedder=embedder, store=store)
        assert ingest(settings, embedder=embedder, store=store).embedded == 0

    def test_does_not_duplicate(self, settings, embedder, store) -> None:
        ingest(settings, embedder=embedder, store=store)
        ingest(settings, embedder=embedder, store=store)
        assert store.count() == 6

    def test_reports_cache_hits(self, settings, embedder, store) -> None:
        ingest(settings, embedder=embedder, store=store)
        assert ingest(settings, embedder=embedder, store=store).cached == 6


class TestSearchable:
    def test_stored_evidence_is_retrievable(self, settings, embedder, store) -> None:
        ingest(settings, embedder=embedder, store=store)
        query = embedder.embed_query(
            "Orchid Labs — Internal search over research documents\n"
            "tags: python, search, elasticsearch, relevance\n"
        )
        hits = store.search(query, limit=6)
        assert len(hits) == 6
        assert all(h.evidence.evidence_id.startswith("ev_") for h in hits)
