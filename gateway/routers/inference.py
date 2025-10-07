from fastapi import APIRouter, Depends
from gateway.schemas.inference_request import InferenceRequest
from gateway.schemas.inference_response import InferenceResponse
from gateway.services.inference_service import InferenceService
from gateway.clients.triton_client import TritonClient, get_triton_client

router = APIRouter()


@router.post("/", response_model=InferenceResponse)
async def infer(
    req: InferenceRequest, client: TritonClient = Depends(get_triton_client)
):
    service = InferenceService(client)
    result = await service.run_inference(req.model_name, req.inputs)
    return result
