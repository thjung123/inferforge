import asyncio
import os
from functools import lru_cache
from typing import List, Any

import torch
from tritonclient.grpc.aio import (
    InferenceServerClient,
    InferInput,
    InferRequestedOutput,
)

from gateway.middlewares.circuit_breaker.manager import breaker_manager
from gateway.middlewares.request_id import request_id_ctx
from gateway.utils.logger import gateway_logger as logger


class TritonClient:
    def __init__(self, max_retries: int = 3, base_delay: float = 0.3):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.triton_breaker = breaker_manager.get("triton")

        self.triton_url = os.getenv("TRITON_URL", "triton:8001")
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
        self, model_name: str, inputs: List[dict[str, Any]], output_names: list[str]
    ) -> dict:
        if not self.triton_breaker.allow_request():
            logger.warning("[TritonBreaker] Circuit open - skipping inference")
            return {"error": "Triton circuit open - skipping inference"}

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

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"[TritonClient] Inference attempt {attempt}/{self.max_retries} | model={model_name}"
                )
                response = await client.infer(
                    model_name=model_name,
                    inputs=infer_inputs,
                    outputs=infer_outputs,
                    parameters=parameters,
                )

                self.triton_breaker.record_success()
                logger.info(f"[TritonClient] Success | model={model_name}")
                return {
                    output.name: response.as_numpy(output.name)
                    for output in response.get_response().outputs
                }

            except Exception as e:
                logger.warning(f"[TritonClient] Attempt {attempt} failed: {e}")
                self.triton_breaker.record_failure()

                if attempt == self.max_retries:
                    logger.error(
                        f"[TritonClient] All retries failed for model={model_name}"
                    )
                    return {"error": str(e)}
                await asyncio.sleep(self.base_delay * (2 ** (attempt - 1)))
        logger.error(
            f"[TritonClient] No response after {self.max_retries} attempts | model={model_name}"
        )
        return {"error": f"No response after {self.max_retries} attempts"}


@lru_cache
def get_triton_client() -> TritonClient:
    return TritonClient()
