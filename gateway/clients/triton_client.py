from functools import lru_cache
from typing import Any

import numpy as np
import torch
from tritonclient.grpc.aio import (
    InferenceServerClient,
    InferInput,
    InferRequestedOutput,
)

from gateway.config import get_settings
from gateway.middlewares.circuit_breaker.manager import breaker_manager
from gateway.middlewares.request_id import request_id_ctx
from gateway.utils.exceptions import TritonCircuitOpenError, TritonInferenceError
from gateway.utils.logger import gateway_logger as logger
from gateway.utils.resilience import async_retry


class TritonClient:
    def __init__(self):
        settings = get_settings()
        self.max_retries = settings.triton_max_retries
        self.base_delay = settings.triton_retry_base_delay
        self.triton_breaker = breaker_manager.get("triton")

        self.triton_url = settings.triton_url
        try:
            self.gpu_enabled = torch.cuda.is_available()
        except Exception:
            self.gpu_enabled = False

        self.triton_enabled = self.gpu_enabled
        self.client = None

    async def _get_client(self):
        if not self.client:
            self.client = InferenceServerClient(url=self.triton_url)
            logger.info(f"[TritonClient] Connected to Triton gRPC at {self.triton_url}")
        return self.client

    async def infer(
        self,
        model_name: str,
        inputs: list[dict[str, Any]],
        output_names: list[str],
    ) -> dict[str, np.ndarray]:
        if not self.triton_breaker.allow_request():
            logger.warning("[TritonBreaker] Circuit open - skipping inference")
            raise TritonCircuitOpenError()

        if not self.triton_enabled:
            raise RuntimeError("Triton not available on this system")

        request_id = request_id_ctx.get()
        client = await self._get_client()

        infer_inputs = []
        for inp in inputs:
            tensor = InferInput(inp["name"], inp["shape"], inp["datatype"])
            tensor.set_data_from_numpy(inp["data"])
            infer_inputs.append(tensor)

        infer_outputs = [InferRequestedOutput(name) for name in output_names]
        parameters = {"request_id": str(request_id)} if request_id else {}

        async def _do_infer():
            response = await client.infer(
                model_name=model_name,
                inputs=infer_inputs,
                outputs=infer_outputs,
                parameters=parameters,
            )
            return {
                output.name: response.as_numpy(output.name)
                for output in response.get_response().outputs
            }

        def _on_retry(attempt: int, exc: Exception):
            logger.warning(
                f"[TritonClient] Attempt {attempt}/{self.max_retries} failed: {exc}"
            )
            self.triton_breaker.record_failure()

        try:
            result = await async_retry(
                _do_infer,
                max_retries=self.max_retries,
                base_delay=self.base_delay,
                on_retry=_on_retry,
            )
            self.triton_breaker.record_success()
            logger.info(f"[TritonClient] Success | model={model_name}")
            return result
        except Exception as e:
            logger.error(
                f"[TritonClient] All retries failed for model={model_name}: {e}"
            )
            raise TritonInferenceError(detail=f"Inference failed for {model_name}: {e}")


@lru_cache
def get_triton_client() -> TritonClient:
    return TritonClient()
