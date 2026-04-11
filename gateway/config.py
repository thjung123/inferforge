from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    triton_url: str = "triton:8001"
    triton_http_url: str = "http://localhost:8000"
    builder_url: str = "http://localhost:8090"
    vllm_primary_url: str = "http://vllm-primary:8100"
    vllm_fallback_url: str = "http://vllm-fallback:8101"
    redis_url: str = "redis://localhost:6379"
    jwt_secret: str = "default_secret"
    api_key_whitelist: list[str] = []

    rate_limit_infer: int = 120
    rate_limit_generate: int = 60
    rate_window: int = 60
    concurrency_limit_infer: int = 64
    concurrency_limit_generate_primary: int = 16
    concurrency_limit_generate_fallback: int = 8
    triton_max_retries: int = 3
    triton_retry_base_delay: float = 0.3
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "lora-adapters"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
