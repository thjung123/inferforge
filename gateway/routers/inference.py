from typing import Annotated
import asyncio
import time

from fastapi import APIRouter, Depends

from gateway.clients.redis_client import get_redis_client
from gateway.config import get_settings
from gateway.schemas.inference_request import InferenceRequest
from gateway.services.embedding_cache import get_cached, set_cached
from gateway.services.inference_manager.dispatcher import InferenceDispatcher
from gateway.services.inference_service import InferenceService, get_inference_service
from gateway.services.model_tiering import ModelTieringService
from gateway.services.tiering_actions import restore, triton_load

router = APIRouter(redirect_slashes=False)

_usage_tasks: set = set()


async def _record_usage(model_name: str) -> None:
    try:
        redis = await get_redis_client()
        await ModelTieringService(redis).record_access(model_name, time.time())
    except Exception:
        pass


async def _ensure_loaded(model_name: str) -> None:
    try:
        redis = await get_redis_client()
        await ModelTieringService(redis).ensure_hot(
            model_name, time.time(), restore=restore, load=triton_load
        )
    except Exception:
        pass


@router.post("", response_model=None)
@router.post("/", response_model=None)
async def infer(
    req: InferenceRequest,
    service: Annotated[InferenceService, Depends(get_inference_service)],
):
    tiering_on = get_settings().enable_tiering

    cached = await get_cached(req.model_name, req.inputs)
    if cached is not None:
        if tiering_on:
            t = asyncio.create_task(_record_usage(req.model_name))
            _usage_tasks.add(t)
            t.add_done_callback(_usage_tasks.discard)
        return cached

    if tiering_on:
        await _ensure_loaded(req.model_name)

    dispatcher = InferenceDispatcher(service)
    result = await dispatcher.run(req.model_name, req.inputs)

    payload = result.model_dump() if hasattr(result, "model_dump") else result
    await set_cached(req.model_name, req.inputs, payload)
    return payload
