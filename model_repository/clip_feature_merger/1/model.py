import json

import numpy as np
import triton_python_backend_utils as pb_utils
from builder.processors.clip.feature_merger import ClipFeatureMerger
from gateway.utils.logger import triton_logger as logger


class TritonPythonModel:
    def initialize(self, args):
        logger.info("Initializing ClipFeatureMerger")
        self.merger = ClipFeatureMerger()

    def execute(self, requests):
        responses = []
        for request in requests:
            params = json.loads(request.parameters())
            request_id = params.get("request_id")
            logger.info(f"[ClipFeatureMerger] req_id={request_id}")

            image_emb = pb_utils.get_input_tensor_by_name(
                request, "image_emb"
            ).as_numpy()
            text_emb = pb_utils.get_input_tensor_by_name(request, "text_emb").as_numpy()
            similarity = self.merger.merge(image_emb, text_emb)
            out = pb_utils.Tensor("similarity", similarity.astype(np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
            logger.info(
                f"[ClipFeatureMerger] Completed req_id={request_id} | shape={similarity.shape}"
            )
        return responses
