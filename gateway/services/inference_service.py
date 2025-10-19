from gateway.clients.triton_client import TritonClient
from gateway.utils.exceptions import InvalidInputError, TritonConnectionError
from gateway.utils.logger import gateway_logger as logger
from typing import List
import time


class InferenceService:
    def __init__(self, client: TritonClient):
        self.client = client

    async def run_inference(self, model_name: str, inputs: List[dict]) -> dict:
        if not inputs:
            logger.warning("Inference failed: empty input")
            raise InvalidInputError("Input data is empty")

        logger.info(f"Starting inference | model={model_name}")
        start_time = time.time()
        try:
            raw_result = await self.client.infer(model_name, inputs)
        except TimeoutError:
            logger.error(f"Triton timeout | model={model_name}")
            raise TritonConnectionError("Triton inference timed out")
        except Exception as e:
            logger.exception(f"Triton call failed | model={model_name} | error={e}")
            raise TritonConnectionError(f"Triton call failed: {e}")

        duration = time.time() - start_time
        logger.info(
            f"Inference completed | model={model_name}, duration={duration:.3f}s, "
            f"result_keys={list(raw_result.keys())}"
        )
        return raw_result
