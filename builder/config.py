from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class BuilderSettings(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"
    model_repository: str = "/models"
    triton_http_url: str = "http://triton:8000"
    max_concurrent_builds: int = 1

    push_to_object_store: bool = False
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    model_bucket: str = "model-repository"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_builder_settings() -> BuilderSettings:
    return BuilderSettings()
