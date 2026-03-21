from typing import cast, Awaitable, Any

from fastapi import Request
from fastapi.responses import JSONResponse
from gateway.clients.redis_client import get_redis_client
from gateway.config import get_settings
from gateway.middlewares.request_id import request_id_ctx
from gateway.utils.logger import gateway_logger as logger


LUA_SCRIPT = """
local current = redis.call("INCR", KEYS[1])
if tonumber(current) == 1 then
  redis.call("EXPIRE", KEYS[1], ARGV[1])
end
return current
"""


async def rate_limiter(request: Request, call_next):
    settings = get_settings()
    client_ip = request.client.host if request.client else "unknown"
    key = f"rate:{client_ip}"
    redis = await get_redis_client()

    try:
        count = await cast(
            Awaitable[Any],
            redis.eval(LUA_SCRIPT, 1, key, str(settings.rate_window)),
        )
        logger.debug(f"[RateLimiter] key={key}, count={count}")
    except Exception as e:
        logger.warning(f"[RateLimiter] Redis unavailable, skipping limit: {e}")
        return await call_next(request)

    if int(count) > settings.rate_limit:
        request_id = request_id_ctx.get()
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "status_code": 429,
                "request_id": str(request_id),
            },
        )

    return await call_next(request)
