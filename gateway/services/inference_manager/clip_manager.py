import numpy as np

from gateway.services.inference_manager.base import BaseManager
from gateway.schemas.triton_models.clip import ClipRequest, ClipResponse


class ClipManager(BaseManager):
    MODEL_NAME = "clip_ensemble"

    async def run(self, image_urls: list[str], texts: list[str]) -> ClipResponse:
        req = ClipRequest(IMAGE_URLS=image_urls, TEXTS=texts)
        inputs = req.to_triton_inputs()
        raw = await self.service.run_inference(self.MODEL_NAME, inputs)
        outputs = {out["name"]: out for out in raw.get("outputs", [])}
        sim_out = outputs.get("similarity")

        if sim_out is None:
            raise ValueError(f"Triton response missing 'similarity' output: {raw}")

        data = sim_out["data"]
        shape = sim_out.get("shape", [])

        similarity = np.array(data, dtype=np.float32).reshape(shape).tolist()
        return ClipResponse(similarity=similarity)
