import time

import numpy as np

from fastapi import Depends

from gateway.clients.triton_client import TritonClient, get_triton_client
from gateway.utils.exceptions import InvalidInputError
from gateway.utils.logger import gateway_logger as logger


class InferenceService:
    def __init__(self, client: TritonClient):
        self.client = client

    async def run_inference(
        self, model_name: str, inputs: list[dict], output_names: list[str]
    ) -> dict[str, np.ndarray]:
        if not inputs:
            logger.warning("Inference failed: empty input")
            raise InvalidInputError("Input data is empty")

        logger.info(f"Starting inference | model={model_name}")
        start_time = time.time()

        raw_result = await self.client.infer(model_name, inputs, output_names)

        duration = time.time() - start_time
        logger.info(
            f"Inference completed | model={model_name}, duration={duration:.3f}s, "
            f"result_keys={list(raw_result.keys())}"
        )
        return raw_result


def get_inference_service(
    client: TritonClient = Depends(get_triton_client),
) -> InferenceService:
    return InferenceService(client)
