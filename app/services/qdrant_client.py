from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.config import settings

_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    """Return a lazily-initialised shared AsyncQdrantClient."""
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
    return _client


async def ensure_collection() -> None:
    """Create the cv_candidates collection if it does not already exist."""
    client = get_client()
    existing = await client.get_collections()
    names = [c.name for c in existing.collections]
    if settings.qdrant_collection not in names:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dims,
                distance=Distance.COSINE,
            ),
        )


async def upsert_chunks(
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
) -> list[str]:
    """
    Upsert chunk vectors into Qdrant.

    Each chunk dict must have:
        - "text": str
        - "metadata": dict  (candidate_id, section, original_filename, ...)

    Returns the list of point ID strings that were upserted.
    """
    client = get_client()
    points: list[PointStruct] = []
    point_ids: list[str] = []

    for chunk, vector in zip(chunks, vectors):
        point_id = str(uuid.uuid4())
        point_ids.append(point_id)
        payload = {"text": chunk["text"], **chunk.get("metadata", {})}
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

    await client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )
    return point_ids
