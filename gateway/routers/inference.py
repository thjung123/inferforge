from fastapi import APIRouter, Depends

from gateway.schemas.inference_request import InferenceRequest
from gateway.services.embedding_cache import get_cached, set_cached
from gateway.services.inference_manager.dispatcher import InferenceDispatcher
from gateway.services.inference_service import InferenceService, get_inference_service

router = APIRouter(redirect_slashes=False)


@router.post("", response_model=None)
@router.post("/", response_model=None)
async def infer(
    req: InferenceRequest,
    service: InferenceService = Depends(get_inference_service),
):
    cached = await get_cached(req.model_name, req.inputs)
    if cached is not None:
        return cached

    dispatcher = InferenceDispatcher(service)
    result = await dispatcher.run(req.model_name, req.inputs)

    await set_cached(req.model_name, req.inputs, result)
    return result
