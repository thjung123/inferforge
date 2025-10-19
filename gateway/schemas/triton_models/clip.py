from pydantic import BaseModel
from typing import List


class ClipRequest(BaseModel):
    IMAGE_URLS: List[str]
    TEXTS: List[str]

    def to_triton_inputs(self) -> list[dict]:
        return [
            {
                "name": "IMAGE_URLS",
                "datatype": "BYTES",
                "shape": [1, len(self.IMAGE_URLS)],
                "data": [self.IMAGE_URLS],
            },
            {
                "name": "TEXTS",
                "datatype": "BYTES",
                "shape": [1, len(self.TEXTS)],
                "data": [self.TEXTS],
            },
        ]


class ClipResponse(BaseModel):
    similarity: list[list[float]]
