import numpy as np
from gateway.services.inference_manager.base import BaseManager
from gateway.schemas.triton_models.bert import BertRequest, BertResponse
from gateway.utils.logger import gateway_logger as logger


class BertManager(BaseManager):
    MODEL_NAME = "bert_ensemble"

    async def run(self, texts: list[str]) -> BertResponse:
        req = BertRequest(TEXTS=texts)
        inputs = req.to_triton_inputs()

        raw = await self.service.run_inference(self.MODEL_NAME, inputs)
        outputs = {out["name"]: out for out in raw.get("outputs", [])}

        bert_out = outputs.get("bert_emb")
        if bert_out is None:
            raise ValueError(f"Triton response missing 'bert_emb' output: {raw}")

        data = bert_out["data"]
        shape = bert_out.get("shape", [])

        bert_emb = np.array(data, dtype=np.float32).reshape(shape).tolist()
        logger.info(f"[Postprocessor] Received BERT embeddings shape={shape}")

        return BertResponse(bert_emb=bert_emb)
