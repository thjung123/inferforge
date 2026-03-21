from pydantic import BaseModel


class RegisterRequest(BaseModel):
    model_type: str
    instance_count: int | None = None


class RegisterResponse(BaseModel):
    job_id: str
    model_name: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    model_name: str
    status: str
    error: str | None = None


class ModelInfo(BaseModel):
    name: str
    state: str
    reason: str | None = None


class ModelListResponse(BaseModel):
    models: list[ModelInfo]


class ModelActionResponse(BaseModel):
    name: str
    action: str
    success: bool
    detail: str | None = None
