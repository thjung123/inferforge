import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from gateway.clients.vllm_client import VLLMClient

logger = logging.getLogger("gateway")


class GenerationService:
    def __init__(self, client: VLLMClient):
        self._client = client

    async def generate(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        logger.info(f"[Generate] model={model} max_tokens={max_tokens}")
        start = time.time()

        result = await self._client.generate(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )

        duration = time.time() - start
        logger.info(f"[Generate] completed in {duration:.3f}s")
        return result

    async def generate_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AsyncIterator[bytes]:
        logger.info(f"[Generate] streaming model={model} max_tokens={max_tokens}")

        async for chunk in self._client.generate_stream(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            yield chunk
