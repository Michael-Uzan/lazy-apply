"""Turning a Role into an EvidencePack.

One search per requirement. Query embeddings go out in a single batched
request -- a role has roughly ten requirements, and one round trip each would
be minutes of waiting on a rate-limited tier.
"""

from __future__ import annotations

from lazyapply_core.domain import EvidencePack, RequirementEvidence, Role
from lazyapply_core.embed.base import Embedder
from lazyapply_core.store.base import VectorStore


def build_pack(role: Role, *, embedder: Embedder, store: VectorStore, top_k: int) -> EvidencePack:
    """Retrieve candidate evidence for every requirement in ``role``."""
    queries = [requirement.text for requirement in role.requirements]
    vectors = embedder.embed_queries(queries)

    return EvidencePack(
        role_id=role.role_id,
        items=tuple(
            RequirementEvidence(
                requirement=requirement,
                hits=store.search(vector, limit=top_k),
            )
            for requirement, vector in zip(role.requirements, vectors, strict=True)
        ),
    )
