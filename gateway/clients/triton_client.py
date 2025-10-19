import asyncio
import os
from functools import lru_cache
from typing import List

import aiohttp
from aiohttp import ClientTimeout
from gateway.middlewares.circuit_breaker.manager import breaker_manager
from gateway.utils.logger import gateway_logger as logger
import torch


class TritonClient:
    def __init__(self, max_retries: int = 3, base_delay: float = 0.3):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.triton_breaker = breaker_manager.get("triton")

        self.triton_url = os.getenv("TRITON_URL", "http://triton:8001")
        try:
            self.gpu_enabled = torch.cuda.is_available()
        except Exception:
            self.gpu_enabled = False
        self.triton_enabled = self.gpu_enabled and self.triton_url.startswith("http")
        logger.info(
            f"[TritonClient INIT] gpu_enabled={self.gpu_enabled}, "
            f"triton_enabled={self.triton_enabled}, "
            f"triton_url={self.triton_url}"
        )

    async def infer(self, model_name: str, inputs: List[dict]) -> dict:
        if not self.triton_breaker.allow_request():
            logger.warning("[TritonBreaker] Circuit open - skipping inference")
            return {"error": "Triton circuit open - skipping inference"}

        if not self.triton_enabled:
            logger.error(
                f"[TritonClient] Triton not available (gpu_enabled={self.gpu_enabled}, url={self.triton_url})"
            )
            raise RuntimeError("Triton not available on this system")

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"[TritonClient] Inference attempt {attempt}/{self.max_retries} | model={model_name}"
                )
                async with aiohttp.ClientSession() as session:
                    url = f"{self.triton_url}/v2/models/{model_name}/infer"
                    payload = {"inputs": inputs}
                    timeout = ClientTimeout(total=10)
                    async with session.post(url, json=payload, timeout=timeout) as resp:
                        if resp.status != 200:
                            raise RuntimeError(f"Triton HTTP {resp.status}")
                        data = await resp.json()
                        logger.info("Response: ", data)
                logger.info(f"[TritonClient] Success | model={model_name}")
                self.triton_breaker.record_success()
                return data

            except Exception as e:
                logger.warning(f"[TritonClient] Attempt {attempt} failed: {e}")
                self.triton_breaker.record_failure()

                if attempt == self.max_retries:
                    logger.error(
                        f"[TritonClient] All {self.max_retries} retries failed for model={model_name}"
                    )
                    return {"error": str(e)}
                await asyncio.sleep(self.base_delay * (2 ** (attempt - 1)))
        return {"error": "unreachable"}


@lru_cache
def get_triton_client() -> TritonClient:
    return TritonClient()
