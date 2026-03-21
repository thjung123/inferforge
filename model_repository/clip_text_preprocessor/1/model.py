import json

import triton_python_backend_utils as pb_utils
from builder.processors.clip.text_preprocessor import ClipTextPreprocessor
from gateway.utils.logger import triton_logger as logger


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        max_length = int(params.get("max_length", {}).get("string_value", "77"))
        logger.info(f"Initializing ClipTextPreprocessor (max_length={max_length})")
        self.processor = ClipTextPreprocessor(max_length=max_length)

    def execute(self, requests):
        responses = []
        for request in requests:
            params = json.loads(request.parameters())
            request_id = params.get("request_id")

            input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXTS")
            texts = [t.decode("utf-8") for t in input_tensor.as_numpy().flatten()]

            logger.info(
                f"[ClipTextPreprocessor] req_id={request_id} | {len(texts)} texts"
            )

            result = self.processor.run(texts)

            tensors = [
                pb_utils.Tensor("INPUT_IDS", result["input_ids"]),
                pb_utils.Tensor("ATTENTION_MASK", result["attention_mask"]),
            ]
            responses.append(pb_utils.InferenceResponse(output_tensors=tensors))
        return responses
