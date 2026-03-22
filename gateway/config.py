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

    rate_limit: int = 20
    rate_window: int = 3
    triton_max_retries: int = 3
    triton_retry_base_delay: float = 0.3

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
