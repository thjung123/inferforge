from gateway.services.preprocess import preprocess_inputs
from gateway.services.postprocess import postprocess_outputs
from gateway.clients.triton_client import TritonClient
from gateway.utils.exceptions import InvalidInputError, TritonConnectionError
from gateway.utils.logger import logger
import time


class InferenceService:
    def __init__(self, client: TritonClient):
        self.client = client

    async def run_inference(self, model_name: str, inputs: dict) -> dict:
        if not inputs:
            logger.warning("Inference failed: empty input")
            raise InvalidInputError("Input data is empty")

        logger.info(
            f"Starting inference | model={model_name}, input_size={len(inputs)}"
        )
        start_time = time.time()

        processed = preprocess_inputs(inputs)

        try:
            raw_result = self.client.infer(model_name, processed)
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

        return postprocess_outputs(raw_result)
