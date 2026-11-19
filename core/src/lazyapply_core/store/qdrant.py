"""Qdrant-backed vector store.

The only module that knows Qdrant exists. Two choices here are worth naming:

- Points are keyed by a UUID derived from the evidence id, so re-running
  ingest replaces rather than duplicates.
- Reads and writes go through an alias, never the collection name. That makes
  a future reindex a pointer swap instead of downtime.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from lazyapply_core.domain import Evidence, ScoredEvidence
from lazyapply_core.embed.base import Vector

#: Fixed namespace so evidence_id -> point id is stable across machines and runs.
_NAMESPACE = uuid.UUID("6f1d8b3a-6a1e-4f9f-9c3f-2b5f1a0d7e42")


def point_id(evidence_id: str) -> str:
    """Deterministic point id. Qdrant accepts UUIDs or integers, not slugs."""
    return str(uuid.uuid5(_NAMESPACE, evidence_id))


class QdrantStore:
    def __init__(self, *, client: QdrantClient, collection: str, alias: str) -> None:
        self._client = client
        self._collection = collection
        self._alias = alias

    def ensure_collection(self, *, dimensions: int) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=dimensions, distance=models.Distance.COSINE
                ),
            )
            # Indexed payload fields can be filtered on without a full scan.
            for field in ("tags", "company", "source_file"):
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

        if not any(a.alias_name == self._alias for a in self._client.get_aliases().aliases):
            self._client.update_collection_aliases(
                change_aliases_operations=[
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(
                            collection_name=self._collection, alias_name=self._alias
                        )
                    )
                ]
            )

    def upsert(self, evidence: Sequence[Evidence], vectors: Sequence[Vector]) -> None:
        points = [
            models.PointStruct(
                id=point_id(item.evidence_id),
                vector=list(vector),
                payload=item.model_dump(mode="json"),
            )
            for item, vector in zip(evidence, vectors, strict=True)
        ]
        if points:
            self._client.upsert(collection_name=self._alias, points=points, wait=True)

    def search(self, vector: Vector, *, limit: int) -> tuple[ScoredEvidence, ...]:
        hits = self._client.query_points(
            collection_name=self._alias,
            query=list(vector),
            limit=limit,
            with_payload=True,
        ).points
        return tuple(
            ScoredEvidence(evidence=_to_evidence(hit.payload), score=hit.score) for hit in hits
        )

    def count(self) -> int:
        return self._client.count(collection_name=self._alias, exact=True).count

    def drop(self) -> None:
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)


def _to_evidence(payload: dict[str, Any] | None) -> Evidence:
    if payload is None:
        raise ValueError("Qdrant returned a point with no payload")
    return Evidence.model_validate(payload)
