from typing import Optional, cast
from redis.asyncio import Redis
from gateway.config import get_settings
from gateway.middlewares.circuit_breaker.manager import breaker_manager
from gateway.utils.logger import logger


class RedisClient:
    _instance: Optional[Redis] = None

    @classmethod
    async def get_instance(cls) -> Redis:
        redis_breaker = breaker_manager.get("redis")

        if not redis_breaker.allow_request():
            logger.warning("[RedisBreaker] Circuit open - skipping Redis init")
            raise ConnectionError("Redis circuit is open")

        if cls._instance is None:
            try:
                settings = get_settings()
                cls._instance = await Redis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                logger.info("[RedisClient] Redis connection established")
            except Exception as e:
                redis_breaker.record_failure()
                logger.error(f"[RedisBreaker] Redis connection failed: {e}")
                raise

        return cls._instance

    @classmethod
    async def close(cls):
        if cls._instance:
            await cls._instance.close()
            logger.info("[RedisClient] Redis connection closed")
            cls._instance = None

    async def incr(self, key: str) -> int:
        redis_breaker = breaker_manager.get("redis")

        if not redis_breaker.allow_request():
            logger.warning("[RedisBreaker] Circuit open - skipping incr")
            raise ConnectionError("Redis circuit open")

        if self._instance is None:
            raise RuntimeError("[RedisClient] Redis instance is not initialized")

        redis = cast(Redis, self._instance)
        result = await redis.incr(key)
        return result
