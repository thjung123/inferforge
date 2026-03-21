from typing import Any, Awaitable, cast

from fastapi import APIRouter, Depends
from starlette.status import HTTP_202_ACCEPTED

from gateway.clients.redis_client import RedisClient
from gateway.schemas.model_management import (
    JobStatusResponse,
    ModelActionResponse,
    ModelInfo,
    ModelListResponse,
    RegisterRequest,
    RegisterResponse,
)
from gateway.services.model_management_service import (
    ModelManagementService,
    get_model_management_service,
)
from gateway.utils.exceptions import BuilderUnavailableError, ModelNotFoundError

router = APIRouter()


@router.post(
    "/register", status_code=HTTP_202_ACCEPTED, response_model=RegisterResponse
)
async def register_model(
    body: RegisterRequest,
    svc: ModelManagementService = Depends(get_model_management_service),
):
    try:
        result = await svc.register(body.model_type, body.instance_count)
    except Exception as exc:
        raise BuilderUnavailableError(detail=str(exc))
    return RegisterResponse(
        job_id=result["job_id"],
        model_name=result["model_name"],
        status=result["status"],
    )


@router.get("", response_model=ModelListResponse)
async def list_models(
    svc: ModelManagementService = Depends(get_model_management_service),
):
    index = await svc.list_models()
    models = [
        ModelInfo(
            name=m.get("name", ""),
            state=m.get("state", "UNKNOWN"),
            reason=m.get("reason", None),
        )
        for m in index
    ]
    return ModelListResponse(models=models)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    redis = await RedisClient.get_instance()
    data = await cast(
        Awaitable[dict[str, Any]],
        redis.hgetall(f"build_job:{job_id}"),
    )
    if not data:
        raise ModelNotFoundError(detail=f"Job {job_id} not found")
    return JobStatusResponse(
        job_id=data.get("job_id", job_id),
        model_name=data.get("model_name", ""),
        status=data.get("status", "unknown"),
        error=data.get("error") or None,
    )


@router.post("/{name}/load", response_model=ModelActionResponse)
async def load_model(
    name: str,
    svc: ModelManagementService = Depends(get_model_management_service),
):
    try:
        await svc.load_model(name)
    except RuntimeError as exc:
        raise ModelNotFoundError(detail=str(exc))
    return ModelActionResponse(name=name, action="load", success=True)


@router.post("/{name}/unload", response_model=ModelActionResponse)
async def unload_model(
    name: str,
    svc: ModelManagementService = Depends(get_model_management_service),
):
    try:
        await svc.unload_model(name)
    except RuntimeError as exc:
        raise ModelNotFoundError(detail=str(exc))
    return ModelActionResponse(name=name, action="unload", success=True)
