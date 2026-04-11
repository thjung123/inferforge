from pydantic import BaseModel, Field


class LoRARegisterRequest(BaseModel):
    name: str = Field(..., description="Adapter name")
    base_model: str = Field(
        ..., description="Base model name (e.g. Qwen/Qwen2.5-7B-Instruct)"
    )
    s3_path: str = Field(
        ..., description="Path in MinIO bucket (e.g. adapters/ko-chat-lora)"
    )


class LoRAAdapterResponse(BaseModel):
    name: str
    base_model: str
    s3_path: str
    version: int
    status: str
