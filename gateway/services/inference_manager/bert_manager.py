import numpy as np
from gateway.services.inference_manager.base import BaseManager
from gateway.schemas.triton_models.bert import BertRequest, BertResponse
from gateway.utils.logger import gateway_logger as logger


class BertManager(BaseManager):
    MODEL_NAME = "bert_ensemble"

    async def run(self, texts: list[str]) -> BertResponse:
        req = BertRequest(TEXTS=texts)
        inputs = req.to_triton_inputs()

        raw = await self.service.run_inference(
            self.MODEL_NAME, inputs, req.OUTPUT_NAMES
        )

        output_name = req.OUTPUT_NAMES[0]
        bert_emb = raw.get(output_name)
        if bert_emb is None:
            raise ValueError(f"Triton response missing 'bert_emb' output: {raw}")

        bert_emb = np.array(bert_emb, dtype=np.float32)
        logger.info(f"[BertManager] Received {output_name} shape={bert_emb.shape}")
        bert_emb_list = bert_emb.tolist()
        return BertResponse(bert_emb=bert_emb_list)
