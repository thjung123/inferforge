from fastapi import APIRouter, Depends

from gateway.schemas.inference_request import InferenceRequest
from gateway.services.inference_manager.dispatcher import InferenceDispatcher
from gateway.services.inference_service import InferenceService, get_inference_service

router = APIRouter(redirect_slashes=False)


@router.post("", response_model=None)
@router.post("/", response_model=None)
async def infer(
    req: InferenceRequest,
    service: InferenceService = Depends(get_inference_service),
):
    dispatcher = InferenceDispatcher(service)
    result = await dispatcher.run(req.model_name, req.inputs)
    return result
