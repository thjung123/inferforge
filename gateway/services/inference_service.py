from gateway.services.preprocess import preprocess_inputs
from gateway.services.postprocess import postprocess_outputs
from gateway.clients.triton_client import TritonClient


class InferenceService:
    def __init__(self, client: TritonClient):
        self.client = client

    async def run_inference(self, model_name: str, inputs: dict) -> dict:
        processed = preprocess_inputs(inputs)
        raw_result = self.client.infer(model_name, processed)
        return postprocess_outputs(raw_result)
