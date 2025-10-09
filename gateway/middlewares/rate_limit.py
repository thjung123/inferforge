from fastapi import Request, HTTPException
from gateway.clients.redis_client import RedisClient

RATE_LIMIT = 100


async def rate_limiter(request: Request, call_next):
    redis = await RedisClient.get_instance()
    client_ip = request.client.host
    key = f"rate:{client_ip}"
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, 60)
    if count > RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return await call_next(request)
