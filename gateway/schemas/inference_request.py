from pydantic import BaseModel, Field
from typing import Dict, Any


class InferenceRequest(BaseModel):
    model_name: str = Field(..., description="Target models name deployed on Triton")
    inputs: Dict[str, Any] = Field(..., description="Input data for inference")
