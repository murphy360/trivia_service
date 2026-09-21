from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_api_key: str = Field(alias="TRIVIA_SERVICE_API_KEY")

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-5", alias="ANTHROPIC_MODEL")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")

    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-flash-latest", alias="GEMINI_MODEL")

    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/trivia.db", alias="DATABASE_URL"
    )

    novelty_similarity_threshold: float = Field(
        default=0.87, alias="NOVELTY_SIMILARITY_THRESHOLD"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
