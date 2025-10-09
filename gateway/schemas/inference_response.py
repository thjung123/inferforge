from pydantic import BaseModel, Field


class InferenceResponse(BaseModel):
    class_: str = Field(alias="class")
    confidence: float
