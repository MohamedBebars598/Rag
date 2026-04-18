from __future__ import annotations

from openai import AsyncOpenAI

from app.config import settings

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """Return a lazily-initialised shared AsyncOpenAI client pointed at OpenRouter."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
    return _client


async def chat_completion(
    messages: list[dict],
    model: str | None = None,
    response_format: dict | None = None,
) -> str:
    """Send a chat completion request via OpenRouter and return the content string."""
    client = get_client()
    kwargs: dict = {
        "model": model or settings.llm_model,
        "messages": messages,
    }
    if response_format:
        kwargs["response_format"] = response_format

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


async def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed a batch of texts via OpenRouter and return a list of float vectors."""
    client = get_client()
    response = await client.embeddings.create(
        model=model or settings.embedding_model,
        input=texts,
    )
    # Preserve input order
    sorted_data = sorted(response.data, key=lambda d: d.index)
    return [item.embedding for item in sorted_data]
