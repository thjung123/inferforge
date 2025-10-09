from functools import lru_cache
from gateway.middlewares.circuit_breaker.manager import breaker_manager
from gateway.utils.logger import logger


class TritonClient:
    def infer(self, model_name: str, inputs: dict) -> dict:
        triton_breaker = breaker_manager.get("triton")

        if not triton_breaker.allow_request():
            logger.warning("[TritonBreaker] Circuit open - skipping inference")
            return {"error": "Triton circuit open - skipping inference"}

        try:
            logger.info(f"[TritonClient] Inference start | model={model_name}")
            return {"class": "cat", "confidence": 0.97}
        except Exception as e:
            triton_breaker.record_failure()
            logger.error(f"[TritonBreaker] Triton inference failed: {e}")
            raise


@lru_cache
def get_triton_client() -> TritonClient:
    return TritonClient()
