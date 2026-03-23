import asyncio
import logging

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.clients.vllm_client import VLLMClient, get_vllm_fallback, get_vllm_primary
from gateway.config import get_settings
from gateway.middlewares.circuit_breaker.manager import breaker_manager
from gateway.schemas.generation import GenerateRequest, GenerateResponse
from gateway.services.generation_service import GenerationService

router = APIRouter()
logger = logging.getLogger("gateway")

_primary_sem: asyncio.Semaphore | None = None
_fallback_sem: asyncio.Semaphore | None = None


def _get_primary_sem() -> asyncio.Semaphore:
    global _primary_sem
    if _primary_sem is None:
        _primary_sem = asyncio.Semaphore(
            get_settings().concurrency_limit_generate_primary
        )
    return _primary_sem


def _get_fallback_sem() -> asyncio.Semaphore:
    global _fallback_sem
    if _fallback_sem is None:
        _fallback_sem = asyncio.Semaphore(
            get_settings().concurrency_limit_generate_fallback
        )
    return _fallback_sem


def _get_primary_service(
    client: VLLMClient = Depends(get_vllm_primary),
) -> GenerationService:
    return GenerationService(client)


def _get_fallback_service(
    client: VLLMClient = Depends(get_vllm_fallback),
) -> GenerationService:
    return GenerationService(client)


async def _try_generate(
    service: GenerationService, model: str, req: GenerateRequest
) -> dict:
    return await service.generate(
        model=model,
        messages=req.messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )


@router.post("", response_model=None)
async def generate(
    req: GenerateRequest,
    primary: GenerationService = Depends(_get_primary_service),
    fallback: GenerationService = Depends(_get_fallback_service),
):
    model = req.model
    if req.lora_adapter:
        model = req.lora_adapter

    vllm_breaker = breaker_manager.get("vllm")
    use_fallback = not vllm_breaker.allow_request()

    if use_fallback:
        logger.warning("[Fallback] vLLM circuit open, routing to fallback")

    # --- Streaming ---
    if req.stream:
        service = fallback if use_fallback else primary
        return StreamingResponse(
            service.generate_stream(
                model=model,
                messages=req.messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            ),
            media_type="text/event-stream",
        )

    # --- Non-streaming: primary → fallback → 503 ---
    if not use_fallback:
        p_sem = _get_primary_sem()
        if not p_sem.locked():
            try:
                async with p_sem:
                    result = await _try_generate(primary, model, req)
                vllm_breaker.record_success()
                return _build_response(result, model)
            except (
                httpx.HTTPStatusError,
                httpx.TimeoutException,
                httpx.ConnectError,
            ) as exc:
                vllm_breaker.record_failure()
                logger.warning(f"[Fallback] Primary failed ({exc}), trying fallback")
        else:
            logger.info("[Fallback] Primary concurrency full, trying fallback")

    # Fallback
    f_sem = _get_fallback_sem()
    if f_sem.locked():
        logger.warning("[Throttle] Both primary and fallback concurrency full")
        return JSONResponse(
            status_code=503,
            content={"error": "Server busy, all models at capacity"},
            headers={"Retry-After": "1"},
        )

    async with f_sem:
        result = await _try_generate(fallback, model, req)
    return _build_response(result, model)


def _build_response(result: dict, model: str) -> GenerateResponse:
    choice = result["choices"][0]["message"]
    return GenerateResponse(
        model=result.get("model", model),
        content=choice["content"],
        usage=result.get("usage"),
    )
