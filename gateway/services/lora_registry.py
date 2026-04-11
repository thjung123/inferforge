import logging
import time
from typing import Awaitable, cast

from redis.asyncio import Redis

from gateway.clients.redis_client import get_redis_client

logger = logging.getLogger("gateway")

_REGISTRY_PREFIX = "lora:adapter:"
_REGISTRY_INDEX = "lora:adapters"


class LoRAAdapter:
    def __init__(
        self,
        name: str,
        base_model: str,
        s3_path: str,
        version: int = 1,
        status: str = "active",
        created_at: float | None = None,
    ):
        self.name = name
        self.base_model = base_model
        self.s3_path = s3_path
        self.version = version
        self.status = status
        self.created_at = created_at or time.time()

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "base_model": self.base_model,
            "s3_path": self.s3_path,
            "version": str(self.version),
            "status": self.status,
            "created_at": str(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "LoRAAdapter":
        return cls(
            name=data["name"],
            base_model=data["base_model"],
            s3_path=data["s3_path"],
            version=int(data.get("version", "1")),
            status=data.get("status", "active"),
            created_at=float(data.get("created_at", "0")),
        )


async def register_adapter(adapter: LoRAAdapter) -> None:
    redis: Redis = await get_redis_client()
    key = f"{_REGISTRY_PREFIX}{adapter.name}"
    mapping: dict[str, str] = adapter.to_dict()
    await cast(
        Awaitable[int],
        redis.hset(key, mapping=mapping),  # type: ignore[arg-type]
    )
    await cast(Awaitable[int], redis.sadd(_REGISTRY_INDEX, adapter.name))
    logger.info(
        f"[LoRA Registry] Registered adapter: {adapter.name} v{adapter.version}"
    )


async def remove_adapter(name: str) -> bool:
    redis: Redis = await get_redis_client()
    key = f"{_REGISTRY_PREFIX}{name}"
    deleted = await cast(Awaitable[int], redis.delete(key))
    await cast(Awaitable[int], redis.srem(_REGISTRY_INDEX, name))
    if deleted:
        logger.info(f"[LoRA Registry] Removed adapter: {name}")
    return deleted > 0


async def get_adapter(name: str) -> LoRAAdapter | None:
    redis: Redis = await get_redis_client()
    key = f"{_REGISTRY_PREFIX}{name}"
    data: dict[str, str] = await cast(Awaitable[dict[str, str]], redis.hgetall(key))
    if not data:
        return None
    return LoRAAdapter.from_dict(data)


async def list_adapters() -> list[LoRAAdapter]:
    redis: Redis = await get_redis_client()
    names: set[str] = await cast(Awaitable[set[str]], redis.smembers(_REGISTRY_INDEX))
    adapters = []
    for name in names:
        adapter = await get_adapter(name)
        if adapter:
            adapters.append(adapter)
    return adapters
