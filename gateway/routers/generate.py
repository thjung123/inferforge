import logging
import time

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.clients.vllm_client import VLLMClient, get_vllm_fallback, get_vllm_primary
from gateway.middlewares.adaptive_concurrency import (
    get_fallback_limiter,
    get_primary_limiter,
)
from gateway.middlewares.circuit_breaker.manager import breaker_manager
from gateway.schemas.generation import GenerateRequest, GenerateResponse
from gateway.services.generation_service import GenerationService

router = APIRouter()
logger = logging.getLogger("gateway")


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
    p_limiter = get_primary_limiter()
    f_limiter = get_fallback_limiter()

    if not use_fallback and p_limiter.is_available():
        acquired = await p_limiter.acquire()
        if acquired:
            start = time.time()
            try:
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
            finally:
                p_limiter.release(time.time() - start)
    elif not use_fallback:
        logger.info(
            f"[Adaptive] Primary at capacity "
            f"(limit={p_limiter.current_limit}, in_flight={p_limiter.in_flight}, "
            f"avg_latency={p_limiter.avg_latency:.3f}s), trying fallback"
        )

    # Fallback
    if not f_limiter.is_available():
        logger.warning(
            f"[Throttle] All at capacity "
            f"(primary={p_limiter.current_limit}, fallback={f_limiter.current_limit})"
        )
        return JSONResponse(
            status_code=503,
            content={"error": "Server busy, all models at capacity"},
            headers={"Retry-After": "1"},
        )

    await f_limiter.acquire()
    start = time.time()
    try:
        result = await _try_generate(fallback, model, req)
    finally:
        f_limiter.release(time.time() - start)

    return _build_response(result, model)


def _build_response(result: dict, model: str) -> GenerateResponse:
    choice = result["choices"][0]["message"]
    return GenerateResponse(
        model=result.get("model", model),
        content=choice["content"],
        usage=result.get("usage"),
    )
