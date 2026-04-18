from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    llm_model: str = "openai/gpt-4.1-mini"
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dims: int = 1536

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "cv_candidates"


settings = Settings()
