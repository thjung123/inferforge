import numpy as np
from gateway.utils.logger import gateway_logger as logger


class ClipFeatureMerger:
    def __init__(self):
        logger.info("[Postprocessor] CLIP Feature Merger initialized")

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norm, 1e-8, None)

    def merge(self, image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
        image_emb = self._normalize(image_emb)
        text_emb = self._normalize(text_emb)
        similarity = np.matmul(image_emb, text_emb.T)
        logger.info(f"[Postprocessor] Computed similarity {similarity.shape}")
        return similarity
