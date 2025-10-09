from fastapi import Request
from fastapi.responses import JSONResponse
from gateway.clients.redis_client import RedisClient
import os


RATE_LIMIT = int(os.getenv("RATE_LIMIT", 20))
RATE_WINDOW = int(os.getenv("RATE_WINDOW", 3))


async def rate_limiter(request: Request, call_next):
    client_ip = request.client.host
    key = f"rate:{client_ip}"

    try:
        count = await RedisClient.incr(key)
        print(f"[RateLimiter] key={key}, count={count}")
    except (ConnectionError, RuntimeError, Exception) as e:
        print(f"[RateLimiter] Redis unavailable, skipping limit: {e}")
        return await call_next(request)

    if count == 1:
        try:
            redis = await RedisClient.get_instance()
            await redis.expire(key, RATE_WINDOW)
        except Exception as e:
            print(f"[RateLimiter] expire() failed: {e}")

    if count > RATE_LIMIT:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    return await call_next(request)
