"""Turning a Role into an EvidencePack.

One search per requirement -- with one exception. Query embeddings go out in a
single batched request: a role has roughly ten requirements, and one round
trip each would be minutes of waiting on a rate-limited tier.

The exception is hard facts. A degree requirement cannot be answered by
searching impact stories, but retrieval has no way to return nothing, so it
hands back its nearest neighbours regardless. Those requirements are routed
past retrieval entirely and judged against profile.json instead -- see
``Requirement.is_hard_fact``.
"""

from __future__ import annotations

from lazyapply_core.domain import (
    EvidencePack,
    Requirement,
    RequirementEvidence,
    Role,
    ScoredEvidence,
)
from lazyapply_core.embed.base import Embedder
from lazyapply_core.store.base import VectorStore


def build_pack(role: Role, *, embedder: Embedder, store: VectorStore, top_k: int) -> EvidencePack:
    """Retrieve candidate evidence for every requirement in ``role``.

    Hard-fact requirements keep their place in the pack -- they are still
    judged and still count toward coverage -- but arrive with no hits, because
    none were asked for.
    """
    searchable = tuple(r for r in role.requirements if not r.is_hard_fact)
    hits = _search_all(searchable, embedder=embedder, store=store, top_k=top_k)

    return EvidencePack(
        role_id=role.role_id,
        items=tuple(
            RequirementEvidence(requirement=requirement, hits=hits.get(requirement.id, ()))
            for requirement in role.requirements
        ),
    )


def _search_all(
    requirements: tuple[Requirement, ...],
    *,
    embedder: Embedder,
    store: VectorStore,
    top_k: int,
) -> dict[str, tuple[ScoredEvidence, ...]]:
    if not requirements:
        return {}

    vectors = embedder.embed_queries([r.text for r in requirements])
    return {
        requirement.id: store.search(vector, limit=top_k)
        for requirement, vector in zip(requirements, vectors, strict=True)
    }
