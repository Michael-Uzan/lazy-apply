"""Per-requirement retrieval: a Role becomes an EvidencePack.

One search per requirement rather than one for the whole posting. That is
what lets a requirement with no nearby evidence be identified as a gap, and
it bounds the context sent to the model.
"""

import pytest

from lazyapply_core.domain import (
    Evidence,
    Requirement,
    RequirementCategory,
    RequirementKind,
    Role,
)
from lazyapply_core.embed.fake import FakeEmbedder
from lazyapply_core.retrieve.retriever import build_pack
from lazyapply_core.store.memory import InMemoryStore

DIMENSIONS = 8


def evidence(slug: str, text: str) -> Evidence:
    return Evidence(
        evidence_id=f"ev_test__{slug}",
        title=slug,
        body=text,
        composed_text=text,
        content_hash="0" * 16,
        source_file="test.md",
    )


def role(*ids: str) -> Role:
    return Role(
        role_id="r1",
        title="Test Role",
        raw_text="...",
        requirements=tuple(
            Requirement(
                id=rid,
                text=f"query for {rid}",
                kind=RequirementKind.MUST_HAVE,
                category=RequirementCategory.SKILL,
            )
            for rid in ids
        ),
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dimensions=DIMENSIONS)


@pytest.fixture
def store(embedder: FakeEmbedder) -> InMemoryStore:
    s = InMemoryStore(collection="c", alias="a")
    s.ensure_collection(dimensions=DIMENSIONS)
    items = tuple(evidence(f"e{i}", f"body {i}") for i in range(5))
    s.upsert(items, embedder.embed_documents([e.composed_text for e in items]))
    return s


class TestShape:
    def test_one_entry_per_requirement(self, embedder, store) -> None:
        pack = build_pack(role("a", "b", "c"), embedder=embedder, store=store, top_k=3)
        assert len(pack.items) == 3

    def test_entries_keep_requirement_order(self, embedder, store) -> None:
        pack = build_pack(role("a", "b", "c"), embedder=embedder, store=store, top_k=3)
        assert [i.requirement.id for i in pack.items] == ["a", "b", "c"]

    def test_carries_the_role_id(self, embedder, store) -> None:
        assert build_pack(role("a"), embedder=embedder, store=store, top_k=3).role_id == "r1"

    def test_respects_top_k(self, embedder, store) -> None:
        pack = build_pack(role("a"), embedder=embedder, store=store, top_k=2)
        assert len(pack.items[0].hits) == 2

    def test_hits_are_ordered_by_score(self, embedder, store) -> None:
        hits = build_pack(role("a"), embedder=embedder, store=store, top_k=5).items[0].hits
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


class TestBatching:
    def test_embeds_every_query_in_one_request(self, embedder, store) -> None:
        """Ten requirements must not mean ten round trips."""
        before = embedder.requests
        build_pack(role(*"abcdefghij"), embedder=embedder, store=store, top_k=3)
        assert embedder.requests == before + 1

    def test_embeds_the_requirement_text_not_its_id(self, embedder, store) -> None:
        pack = build_pack(role("a"), embedder=embedder, store=store, top_k=1)
        expected = embedder.embed_query("query for a")
        direct = store.search(expected, limit=1)
        assert pack.items[0].hits[0].evidence == direct[0].evidence


class TestDeduplication:
    def test_unique_evidence_appears_once(self, embedder, store) -> None:
        """The same block is retrieved for several requirements; the prompt
        should carry it once and refer to it by id."""
        pack = build_pack(role("a", "b", "c"), embedder=embedder, store=store, top_k=5)
        unique = pack.unique_evidence()
        assert len(unique) == 5
        assert len({e.evidence_id for e in unique}) == 5

    def test_unique_evidence_is_ordered_deterministically(self, embedder, store) -> None:
        pack = build_pack(role("a", "b"), embedder=embedder, store=store, top_k=5)
        assert [e.evidence_id for e in pack.unique_evidence()] == sorted(
            e.evidence_id for e in pack.unique_evidence()
        )


class TestBestScore:
    def test_reports_the_best_score_per_requirement(self, embedder, store) -> None:
        """Recorded for threshold tuning in slice 9."""
        pack = build_pack(role("a"), embedder=embedder, store=store, top_k=3)
        assert pack.items[0].best_score == pack.items[0].hits[0].score

    def test_best_score_is_none_when_nothing_was_retrieved(self, embedder) -> None:
        empty = InMemoryStore(collection="c", alias="a")
        empty.ensure_collection(dimensions=DIMENSIONS)
        pack = build_pack(role("a"), embedder=embedder, store=empty, top_k=3)
        assert pack.items[0].hits == ()
        assert pack.items[0].best_score is None
