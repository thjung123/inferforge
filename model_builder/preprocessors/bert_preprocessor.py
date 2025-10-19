import numpy as np
from typing import List, Dict
from transformers import BertTokenizerFast
from gateway.utils.logger import gateway_logger as logger


class BertPreprocessor:
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        logger.info(f"[Preprocessor] Initializing BERT tokenizer ({model_name})")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.max_length = max_length

    def run(self, texts: List[str]) -> Dict[str, np.ndarray]:
        logger.info(f"[Preprocessor] Processing {len(texts)} BERT text inputs")

        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)
        token_type_ids = enc["token_type_ids"].astype(np.int64)

        logger.info(f"[Preprocessor] Done. Shape: {input_ids.shape}")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
