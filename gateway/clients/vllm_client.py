import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from gateway.config import get_settings

logger = logging.getLogger("gateway")

_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)


class VLLMClient:
    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=_TIMEOUT)

    async def generate(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        resp = await self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def generate_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AsyncIterator[bytes]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        async with self._client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    yield (line + "\n").encode()

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self._client.aclose()
        logger.info(f"[vLLM] Connection closed: {self._base_url}")


_primary: VLLMClient | None = None
_fallback: VLLMClient | None = None


def get_vllm_primary() -> VLLMClient:
    global _primary
    if _primary is None:
        settings = get_settings()
        _primary = VLLMClient(settings.vllm_primary_url)
    return _primary


def get_vllm_fallback() -> VLLMClient:
    global _fallback
    if _fallback is None:
        settings = get_settings()
        _fallback = VLLMClient(settings.vllm_fallback_url)
    return _fallback
