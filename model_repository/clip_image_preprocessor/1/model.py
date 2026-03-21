import json

import triton_python_backend_utils as pb_utils
from builder.processors.clip.image_preprocessor import ClipImagePreprocessor
from gateway.utils.logger import triton_logger as logger


class TritonPythonModel:
    def initialize(self, args):
        logger.info("Initializing ClipImagePreprocessor")
        self.processor = ClipImagePreprocessor()

    async def execute(self, requests):
        responses = []
        for request in requests:
            params = json.loads(request.parameters())
            request_id = params.get("request_id")

            input_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_URLS")
            urls = [u.decode("utf-8") for u in input_tensor.as_numpy().flatten()]
            logger.info(
                f"[ClipImagePreprocessor] req_id={request_id} | {len(urls)} image URLs"
            )

            batch = await self.processor.run(urls)

            out_tensor = pb_utils.Tensor("IMAGE_TENSOR", batch)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses
