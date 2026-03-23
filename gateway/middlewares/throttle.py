import asyncio
import time
from typing import Any, Awaitable, cast

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from gateway.clients.redis_client import get_redis_client
from gateway.config import get_settings
from gateway.middlewares.request_id import request_id_ctx
from gateway.utils.logger import gateway_logger as logger

# Sliding window rate limit via Redis sorted set
_LUA_SLIDING_WINDOW = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])

redis.call("ZREMRANGEBYSCORE", key, 0, now - window)
local count = redis.call("ZCARD", key)

if count < limit then
    redis.call("ZADD", key, now, now .. ":" .. math.random(1, 1000000))
    redis.call("EXPIRE", key, window)
    return {count + 1, limit, 0}
else
    local oldest = redis.call("ZRANGE", key, 0, 0, "WITHSCORES")
    local retry_after = 0
    if #oldest >= 2 then
        retry_after = math.ceil(tonumber(oldest[2]) + window - now)
    end
    return {count, limit, retry_after}
end
"""

# Concurrency semaphores per endpoint
_semaphores: dict[str, asyncio.Semaphore] = {}

_THROTTLE_PATHS = {"/infer", "/generate"}


def _get_endpoint(path: str) -> str | None:
    for p in _THROTTLE_PATHS:
        if path.startswith(p):
            return p.lstrip("/")
    return None


def _get_semaphore(endpoint: str) -> asyncio.Semaphore:
    if endpoint not in _semaphores:
        settings = get_settings()
        _semaphores[endpoint] = asyncio.Semaphore(settings.concurrency_limit_infer)
    return _semaphores[endpoint]


def _get_rate_config(endpoint: str) -> tuple[int, int]:
    settings = get_settings()
    if endpoint == "generate":
        return settings.rate_limit_generate, settings.rate_window
    return settings.rate_limit_infer, settings.rate_window


async def throttle_middleware(request: Request, call_next) -> Response:
    endpoint = _get_endpoint(request.url.path)
    if endpoint is None:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    rate_limit, window = _get_rate_config(endpoint)

    # --- Sliding window rate limit ---
    try:
        redis = await get_redis_client()
        key = f"throttle:{endpoint}:{client_ip}"
        now = time.time()

        result = await cast(
            Awaitable[list[Any]],
            redis.eval(
                _LUA_SLIDING_WINDOW, 1, key, str(now), str(window), str(rate_limit)
            ),
        )
        current, limit, retry_after = int(result[0]), int(result[1]), int(result[2])
        remaining = max(0, limit - current)

    except Exception as e:
        logger.warning(f"[Throttle] Redis unavailable, skipping rate limit: {e}")
        remaining = -1
        limit = rate_limit
        retry_after = 0
        current = 0

    if current > limit:
        request_id = request_id_ctx.get()
        logger.warning(
            f"[Throttle] Rate limit exceeded: {endpoint} client={client_ip} "
            f"count={current}/{limit}"
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "endpoint": endpoint,
                "request_id": str(request_id),
            },
            headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Window": str(window),
                "Retry-After": str(retry_after),
            },
        )

    # /generate manages its own concurrency (primary → fallback → 503)
    if endpoint == "generate":
        response = await call_next(request)
    else:
        sem = _get_semaphore(endpoint)
        if sem.locked():
            logger.warning(
                f"[Throttle] Concurrency limit reached: {endpoint} client={client_ip}"
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Server busy, too many concurrent requests",
                    "endpoint": endpoint,
                },
                headers={"Retry-After": "1"},
            )
        async with sem:
            response = await call_next(request)

    if remaining >= 0:
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(window)

    return response
