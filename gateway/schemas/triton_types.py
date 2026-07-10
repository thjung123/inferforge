from pydantic import BaseModel
from typing import Any


class TritonInferRequest(BaseModel):
    model_name: str
    inputs: dict[str, Any]


class TritonInferResponse(BaseModel):
    raw_outputs: dict[str, Any]
