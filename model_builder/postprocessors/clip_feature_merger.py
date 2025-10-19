import numpy as np
from gateway.utils.logger import model_builder_logger as logger


class ClipFeatureMerger:
    def __init__(self):
        logger.info("[Postprocessor] CLIP Feature Merger initialized")

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norm, 1e-8, None)

    def merge(
        self, image_emb: np.ndarray, text_emb: np.ndarray, temperature: float = 0.07
    ) -> np.ndarray:
        image_emb = self._normalize(image_emb)
        text_emb = self._normalize(text_emb)
        similarity = np.matmul(image_emb, text_emb.T) / temperature
        exp_sim = np.exp(similarity - np.max(similarity, axis=1, keepdims=True))
        probs = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
        logger.info(
            f"[Postprocessor] Computed softmax similarity (temp={temperature}) {probs.shape}"
        )
        return probs.astype(np.float32)
