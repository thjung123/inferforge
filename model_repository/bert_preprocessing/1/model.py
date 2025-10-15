import triton_python_backend_utils as pb_utils
from models.preprocessors.bert_preprocessor import BertPreprocessor
from gateway.utils.logger import logger


class TritonPythonModel:
    def initialize(self, args):
        logger.info("[Triton] Initializing BertPreprocessor")
        self.processor = BertPreprocessor()

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXTS")
            texts = [t.decode("utf-8") for t in input_tensor.as_numpy().flatten()]
            logger.info(f"[Triton] Received {len(texts)} BERT text inputs")

            result = self.processor.run(texts)

            tensors = [
                pb_utils.Tensor("INPUT_IDS", result["input_ids"]),
                pb_utils.Tensor("ATTENTION_MASK", result["attention_mask"]),
                pb_utils.Tensor("TOKEN_TYPE_IDS", result["token_type_ids"]),
            ]
            responses.append(pb_utils.InferenceResponse(output_tensors=tensors))
        return responses
