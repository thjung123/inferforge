import logging

import numpy as np

logger = logging.getLogger("builder")


class BertMeanPooler:
    """Masked mean pooling over BERT's last_hidden_state → sentence embedding.

    BERT's ``pooled_output`` ([CLS] passed through a dense + tanh layer) is
    trained for the next-sentence-prediction objective and is a poor semantic
    embedding (see Sentence-BERT, and HuggingFace's own pooler_output docs).
    Masked mean pooling over token hidden states — averaging only the non-pad
    tokens — is the standard pooling used by sentence-transformers and gives a
    far better embedding for cosine-similarity search.
    """

    def run(
        self, last_hidden_state: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        mask = attention_mask.astype(np.float32)[:, :, None]
        summed = np.sum(last_hidden_state * mask, axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        emb = (summed / counts).astype(np.float32)
        logger.info(f"[Postprocessor] Mean-pooled embedding {emb.shape}")
        return emb
