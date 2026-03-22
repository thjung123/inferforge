from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from gateway.clients.vllm_client import VLLMClient, get_vllm_primary
from gateway.schemas.generation import GenerateRequest, GenerateResponse
from gateway.services.generation_service import GenerationService

router = APIRouter()


def _get_generation_service(
    client: VLLMClient = Depends(get_vllm_primary),
) -> GenerationService:
    return GenerationService(client)


@router.post("", response_model=None)
async def generate(
    req: GenerateRequest,
    service: GenerationService = Depends(_get_generation_service),
):
    model = req.model
    if req.lora_adapter:
        model = req.lora_adapter

    if req.stream:
        return StreamingResponse(
            service.generate_stream(
                model=model,
                messages=req.messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            ),
            media_type="text/event-stream",
        )

    result = await service.generate(
        model=model,
        messages=req.messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )

    choice = result["choices"][0]["message"]
    return GenerateResponse(
        model=result.get("model", model),
        content=choice["content"],
        usage=result.get("usage"),
    )
