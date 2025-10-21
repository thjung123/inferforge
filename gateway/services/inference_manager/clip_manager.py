import numpy as np
from gateway.services.inference_manager.base import BaseManager
from gateway.schemas.triton_models.clip import ClipRequest, ClipResponse
from gateway.utils.logger import gateway_logger as logger


class ClipManager(BaseManager):
    MODEL_NAME = "clip_ensemble"

    async def run(self, image_urls: list[str], texts: list[str]) -> ClipResponse:
        req = ClipRequest(IMAGE_URLS=image_urls, TEXTS=texts)
        inputs = req.to_triton_inputs()

        raw = await self.service.run_inference(
            self.MODEL_NAME, inputs, req.OUTPUT_NAMES
        )
        output_name = req.OUTPUT_NAMES[0]
        similarity = raw.get(output_name)

        if similarity is None:
            raise ValueError(f"Triton response missing 'similarity' output: {raw}")

        similarity = np.array(similarity, dtype=np.float32)
        logger.info(
            f"[ClipManager] Received similarity matrix shape={similarity.shape}"
        )

        similarity_list = similarity.tolist()
        return ClipResponse(similarity=similarity_list)
