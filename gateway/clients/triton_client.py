import asyncio
from functools import lru_cache
from gateway.middlewares.circuit_breaker.manager import breaker_manager
from gateway.utils.logger import logger


class TritonClient:
    def __init__(self, max_retries: int = 3, base_delay: float = 0.3):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.triton_breaker = breaker_manager.get("triton")

    async def infer(self, model_name: str, inputs: dict) -> dict:
        if not self.triton_breaker.allow_request():
            logger.warning("[TritonBreaker] Circuit open - skipping inference")
            return {"error": "Triton circuit open - skipping inference"}

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"[TritonClient] Inference attempt {attempt}/{self.max_retries} | model={model_name}"
                )
                result = {"class": "cat", "confidence": 0.97}
                logger.info(f"[TritonClient] Success | model={model_name}")
                self.triton_breaker.record_success()
                return result

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
