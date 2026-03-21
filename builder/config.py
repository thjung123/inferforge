from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class BuilderSettings(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"
    model_repository: str = "/models"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_builder_settings() -> BuilderSettings:
    return BuilderSettings()
