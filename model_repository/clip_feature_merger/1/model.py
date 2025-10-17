import numpy as np
import triton_python_backend_utils as pb_utils
from model_builder.postprocessors.clip_feature_merger import ClipFeatureMerger


class TritonPythonModel:
    def initialize(self, args):
        self.merger = ClipFeatureMerger()

    def execute(self, requests):
        responses = []
        for request in requests:
            image_emb = pb_utils.get_input_tensor_by_name(
                request, "image_emb"
            ).as_numpy()
            text_emb = pb_utils.get_input_tensor_by_name(request, "text_emb").as_numpy()
            similarity = self.merger.merge(image_emb, text_emb)
            out = pb_utils.Tensor("similarity", similarity.astype(np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses
