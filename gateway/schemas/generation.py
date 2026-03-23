from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    model: str = Field(
        default="default",
        description="Model name or 'default' to use primary",
    )
    messages: list[dict[str, str]] = Field(
        ..., description="Chat messages in OpenAI format"
    )
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False)
    lora_adapter: str | None = Field(
        default=None, description="LoRA adapter name (if multi-LoRA enabled)"
    )


class GenerateResponse(BaseModel):
    model: str
    content: str
    usage: dict[str, int] | None = None
