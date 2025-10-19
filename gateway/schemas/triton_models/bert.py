from pydantic import BaseModel
from typing import List


class BertRequest(BaseModel):
    TEXTS: List[str]

    def to_triton_inputs(self) -> list[dict]:
        return [
            {
                "name": "TEXTS",
                "datatype": "BYTES",
                "shape": [1, len(self.TEXTS)],
                "data": self.TEXTS,
            }
        ]


class BertResponse(BaseModel):
    bert_emb: list[list[float]]
