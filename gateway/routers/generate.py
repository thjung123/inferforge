import time

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.clients.vllm_client import VLLMClient, get_vllm_fallback, get_vllm_primary
from gateway.config import get_settings
from gateway.middlewares.adaptive_concurrency import (
    get_fallback_limiter,
    get_primary_limiter,
)
from gateway.middlewares.circuit_breaker.manager import breaker_manager
from gateway.schemas.generation import GenerateRequest, GenerateResponse
from gateway.services.generation_service import GenerationService
from gateway.utils.logger import gateway_logger as logger

router = APIRouter()


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
    settings = get_settings()
    if req.lora_adapter:
        primary_model = req.lora_adapter
        fallback_model = settings.vllm_fallback_model
    elif req.model and req.model != "default":
        primary_model = fallback_model = req.model
    else:
        primary_model = settings.vllm_primary_model
        fallback_model = settings.vllm_fallback_model

    vllm_breaker = breaker_manager.get("vllm")
    use_fallback = not vllm_breaker.allow_request()

    if use_fallback:
        logger.warning("[Fallback] vLLM circuit open, routing to fallback")

    if req.stream:
        service = fallback if use_fallback else primary
        stream_model = fallback_model if use_fallback else primary_model
        return StreamingResponse(
            service.generate_stream(
                model=stream_model,
                messages=req.messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            ),
            media_type="text/event-stream",
        )

    p_limiter = get_primary_limiter()
    f_limiter = get_fallback_limiter()

    if not use_fallback and p_limiter.is_available():
        acquired = await p_limiter.acquire()
        if acquired:
            start = time.time()
            try:
                result = await _try_generate(primary, primary_model, req)
                vllm_breaker.record_success()
                return _build_response(result, primary_model)
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
        result = await _try_generate(fallback, fallback_model, req)
    except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as exc:
        vllm_breaker.record_failure()
        logger.error(f"[Fallback] Fallback model failed ({exc})")
        return JSONResponse(
            status_code=503,
            content={"error": "All models failed to generate"},
            headers={"Retry-After": "1"},
        )
    finally:
        f_limiter.release(time.time() - start)

    return _build_response(result, fallback_model)


def _build_response(result: dict, model: str) -> GenerateResponse:
    choice = result["choices"][0]["message"]
    return GenerateResponse(
        model=result.get("model", model),
        content=choice["content"],
        usage=result.get("usage"),
    )
