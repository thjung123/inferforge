from pydantic import BaseModel
from typing import Any, Dict


class TritonInferRequest(BaseModel):
    model_name: str
    inputs: Dict[str, Any]


class TritonInferResponse(BaseModel):
    raw_outputs: Dict[str, Any]
