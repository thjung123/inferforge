import triton_python_backend_utils as pb_utils
from model_builder.preprocessors.clip_image_preprocessor import ClipImagePreprocessor
from gateway.utils.logger import logger


class TritonPythonModel:
    def initialize(self, args):
        logger.info("[Triton] Initializing ClipImagePreprocessor")
        self.processor = ClipImagePreprocessor()

    async def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_URLS")
            urls = [u.decode("utf-8") for u in input_tensor.as_numpy().flatten()]
            logger.info(f"[Triton] Received {len(urls)} image URLs")

            batch = await self.processor.run(urls)

            out_tensor = pb_utils.Tensor("IMAGE_TENSOR", batch)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses
