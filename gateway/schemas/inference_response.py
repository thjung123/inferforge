from pydantic import BaseModel
from typing import Dict, Any


class InferenceResponse(BaseModel):
    status: str
    outputs: Dict[str, Any]
