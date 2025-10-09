from typing import cast, Awaitable, Any

from fastapi import Request
from fastapi.responses import JSONResponse
from gateway.clients.redis_client import RedisClient
import os


RATE_LIMIT = int(os.getenv("RATE_LIMIT", 20))
RATE_WINDOW = int(os.getenv("RATE_WINDOW", 3))


LUA_SCRIPT = """
local LUA_SCRIPT = redis.call("INCR", KEYS[1])
if tonumber(current) == 1 then
  redis.call("EXPIRE", KEYS[1], ARGV[1])
end
return current
"""


async def rate_limiter(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    key = f"rate:{client_ip}"
    redis = await RedisClient.get_instance()

    try:
        count = await cast(
            Awaitable[Any], redis.eval(LUA_SCRIPT, 1, key, str(RATE_WINDOW))
        )
        print(f"[RateLimiter] key={key}, count={count}")
    except Exception as e:
        print(f"[RateLimiter] Redis unavailable, skipping limit: {e}")
        return await call_next(request)

    if int(count) > RATE_LIMIT:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    return await call_next(request)
