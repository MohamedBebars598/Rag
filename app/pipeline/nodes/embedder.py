from __future__ import annotations

from app.pipeline.schemas import PipelineState
from app.services.openrouter import embed_texts
from app.services.qdrant_client import ensure_collection, upsert_chunks


async def embed_node(state: PipelineState) -> dict:
    """Node 3: Embed chunks via OpenRouter and store vectors in Qdrant."""
    chunks = state["chunks"]

    if not chunks:
        return {"stored_ids": []}

    texts = [chunk["text"] for chunk in chunks]

    vectors = await embed_texts(texts)

    await ensure_collection()

    stored_ids = await upsert_chunks(chunks=chunks, vectors=vectors)

    return {"stored_ids": stored_ids}
