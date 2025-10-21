import numpy as np
from pydantic import BaseModel
from typing import List, ClassVar


class ClipRequest(BaseModel):
    IMAGE_URLS: List[str]
    TEXTS: List[str]
    OUTPUT_NAMES: ClassVar[list[str]] = ["similarity"]

    def to_triton_inputs(self) -> list[dict]:
        image_array = np.array([self.IMAGE_URLS], dtype=object)
        text_array = np.array([self.TEXTS], dtype=object)

        return [
            {
                "name": "IMAGE_URLS",
                "datatype": "BYTES",
                "shape": list(image_array.shape),
                "data": image_array,
            },
            {
                "name": "TEXTS",
                "datatype": "BYTES",
                "shape": list(text_array.shape),
                "data": text_array,
            },
        ]


class ClipResponse(BaseModel):
    similarity: list[list[float]]
