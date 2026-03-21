from enum import Enum
from pydantic import BaseModel


class JobState(str, Enum):
    PENDING = "pending"
    BUILDING_ONNX = "building_onnx"
    BUILDING_TRT = "building_trt"
    GENERATING_CONFIG = "generating_config"
    READY = "ready"
    FAILED = "failed"


class BuildRequest(BaseModel):
    model_type: str


class BuildResponse(BaseModel):
    job_id: str
    model_name: str
    status: JobState


class JobStatus(BaseModel):
    job_id: str
    model_name: str
    status: JobState
    error: str | None = None
