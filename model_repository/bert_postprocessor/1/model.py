import json

import numpy as np
import triton_python_backend_utils as pb_utils
from builder.processors.bert.postprocessor import BertMeanPooler
from common.logger import triton_logger as logger


class TritonPythonModel:
    def initialize(self, args):
        logger.info("Initializing BertMeanPooler")
        self.pooler = BertMeanPooler()

    def execute(self, requests):
        responses = []
        for request in requests:
            params = json.loads(request.parameters())
            request_id = params.get("request_id")

            last_hidden_state = pb_utils.get_input_tensor_by_name(
                request, "last_hidden_state"
            ).as_numpy()
            attention_mask = pb_utils.get_input_tensor_by_name(
                request, "attention_mask"
            ).as_numpy()

            emb = self.pooler.run(last_hidden_state, attention_mask)

            out = pb_utils.Tensor("bert_emb", emb.astype(np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
            logger.info(
                f"[BertMeanPooler] Completed req_id={request_id} | shape={emb.shape}"
            )
        return responses
