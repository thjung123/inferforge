import hashlib
import json
import logging
from typing import Any, Awaitable, cast

from redis.asyncio import Redis

from gateway.clients.redis_client import get_redis_client

logger = logging.getLogger("gateway")

_CACHE_PREFIX = "emb_cache:"
_DEFAULT_TTL = 3600  # 1 hour


def _make_key(model_name: str, inputs: dict[str, Any]) -> str:
    raw = json.dumps({"model": model_name, "inputs": inputs}, sort_keys=True)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{_CACHE_PREFIX}{model_name}:{digest}"


async def get_cached(model_name: str, inputs: dict[str, Any]) -> dict[str, Any] | None:
    try:
        redis: Redis = await get_redis_client()
        key = _make_key(model_name, inputs)
        data = await cast(Awaitable[str | None], redis.get(key))
        if data is not None:
            logger.info(f"[Cache] HIT {key}")
            return json.loads(data)
    except Exception as e:
        logger.warning(f"[Cache] Read error: {e}")
    return None


async def set_cached(
    model_name: str,
    inputs: dict[str, Any],
    result: dict[str, Any],
    ttl: int = _DEFAULT_TTL,
) -> None:
    try:
        redis: Redis = await get_redis_client()
        key = _make_key(model_name, inputs)
        await cast(
            Awaitable[bool | None],
            redis.set(key, json.dumps(result), ex=ttl),
        )
        logger.info(f"[Cache] SET {key} (ttl={ttl}s)")
    except Exception as e:
        logger.warning(f"[Cache] Write error: {e}")
