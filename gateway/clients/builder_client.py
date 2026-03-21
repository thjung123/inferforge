from functools import lru_cache

import httpx

from gateway.config import get_settings
from gateway.utils.logger import gateway_logger as logger


class BuilderClient:
    def __init__(self):
        settings = get_settings()
        self._base_url = settings.builder_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)

    async def build(self, model_type: str, instance_count: int | None = None) -> dict:
        payload: dict = {"model_type": model_type}
        if instance_count is not None:
            payload["instance_count"] = instance_count
        resp = await self._client.post("/build", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self._client.aclose()
        logger.info("[BuilderClient] Connection closed")


@lru_cache
def get_builder_client() -> BuilderClient:
    return BuilderClient()
