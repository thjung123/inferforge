import logging
from typing import Dict, List

import numpy as np
from transformers import CLIPTokenizerFast

logger = logging.getLogger("builder")


class ClipTextPreprocessor:
    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch32", max_length: int = 77
    ):
        logger.info(f"[Preprocessor] Initializing CLIP text tokenizer ({model_name})")
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.max_length = max_length

    def run(self, texts: List[str]) -> Dict[str, np.ndarray]:
        logger.info(f"[Preprocessor] Processing {len(texts)} text inputs")

        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        input_ids = enc["input_ids"].astype(np.int32)
        attention_mask = enc["attention_mask"].astype(np.int32)

        logger.info(f"[Preprocessor] Done. Shape: input_ids={input_ids.shape}")
        return {"input_ids": input_ids, "attention_mask": attention_mask}
