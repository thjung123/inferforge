import numpy as np
from pydantic import BaseModel
from typing import List, ClassVar


class BertRequest(BaseModel):
    TEXTS: List[str]
    OUTPUT_NAMES: ClassVar[list[str]] = ["bert_emb"]

    def to_triton_inputs(self) -> list[dict]:
        text_array = np.array([self.TEXTS], dtype=object)
        return [
            {
                "name": "TEXTS",
                "datatype": "BYTES",
                "shape": list(text_array.shape),
                "data": text_array,
            }
        ]


class BertResponse(BaseModel):
    bert_emb: list[list[float]]
