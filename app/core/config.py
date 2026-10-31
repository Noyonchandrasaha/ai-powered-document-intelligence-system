from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="APP_",
        extra="ignore",
    )

    name: str = "Production API"
    version: str = "0.1.0"
    service_name: str = "api"
    environment: Literal["local", "test", "staging", "production"] = "local"
    debug: bool = False
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_database: str = "document_intelligence"
    openai_api_key: str | None = Field(default=None, repr=False)
    openai_model: str = "gpt-5.5"
    openai_embedding_model: str = "text-embedding-3-small"
    semantic_search_enabled: bool = False
    google_vision_credentials_file: str = "Vision-api.text"
    allowed_origins: list[str] = Field(default_factory=list)
    docs_enabled: bool = False
    allowed_hosts: list[str] = Field(default_factory=lambda: ["*"])

    @field_validator("api_prefix")
    @classmethod
    def normalize_api_prefix(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("api_prefix cannot be empty")
        return "/" + value.strip("/")

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper()

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def app_name(self) -> str:
        return self.name

    @model_validator(mode="after")
    def validate_production_hosts(self) -> "Settings":
        if self.is_production and "*" in self.allowed_hosts:
            raise ValueError("APP_ALLOWED_HOSTS must be restricted in production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
