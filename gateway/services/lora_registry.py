import json
import time
from functools import lru_cache
from typing import cast
from collections.abc import Awaitable

from redis.asyncio import Redis

from gateway.clients.redis_client import get_redis_client
from gateway.utils.logger import gateway_logger as logger

_REGISTRY_PREFIX = "lora:adapter:"
_REGISTRY_INDEX = "lora:adapters"
_REGISTRY_CHANNEL = "lora:events"


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


class LoRARegistryService:
    async def _publish_event(self, redis: Redis, event: dict) -> None:
        try:
            await redis.publish(_REGISTRY_CHANNEL, json.dumps(event))
        except Exception as e:
            logger.warning(f"[LoRA Registry] event publish failed: {e}")

    async def register_adapter(self, adapter: LoRAAdapter) -> None:
        redis: Redis = await get_redis_client()
        key = f"{_REGISTRY_PREFIX}{adapter.name}"
        mapping: dict[str, str] = adapter.to_dict()
        await cast(
            Awaitable[int],
            redis.hset(key, mapping=mapping),  # type: ignore[arg-type]
        )
        await cast(Awaitable[int], redis.sadd(_REGISTRY_INDEX, adapter.name))
        await self._publish_event(
            redis,
            {"action": "upsert", "name": adapter.name, "version": adapter.version},
        )
        logger.info(
            f"[LoRA Registry] Registered adapter: {adapter.name} v{adapter.version}"
        )

    async def register_adapter_atomic(
        self, name: str, base_model: str, s3_path: str
    ) -> LoRAAdapter:
        """Register with an atomic version bump (HINCRBY)."""
        redis: Redis = await get_redis_client()
        key = f"{_REGISTRY_PREFIX}{name}"
        version = int(await cast(Awaitable[int], redis.hincrby(key, "version", 1)))
        adapter = LoRAAdapter(
            name=name, base_model=base_model, s3_path=s3_path, version=version
        )
        mapping = {k: v for k, v in adapter.to_dict().items() if k != "version"}
        await cast(Awaitable[int], redis.hset(key, mapping=mapping))  # type: ignore[arg-type]
        await cast(Awaitable[int], redis.sadd(_REGISTRY_INDEX, name))
        await self._publish_event(
            redis, {"action": "upsert", "name": name, "version": version}
        )
        logger.info(f"[LoRA Registry] Registered adapter: {name} v{version}")
        return adapter

    async def remove_adapter(self, name: str) -> bool:
        redis: Redis = await get_redis_client()
        key = f"{_REGISTRY_PREFIX}{name}"
        deleted = await cast(Awaitable[int], redis.delete(key))
        await cast(Awaitable[int], redis.srem(_REGISTRY_INDEX, name))
        await self._publish_event(redis, {"action": "remove", "name": name})
        if deleted:
            logger.info(f"[LoRA Registry] Removed adapter: {name}")
        return deleted > 0

    async def get_adapter(self, name: str) -> LoRAAdapter | None:
        redis: Redis = await get_redis_client()
        key = f"{_REGISTRY_PREFIX}{name}"
        data: dict[str, str] = await cast(Awaitable[dict[str, str]], redis.hgetall(key))
        if not data:
            return None
        return LoRAAdapter.from_dict(data)

    async def list_adapters(self) -> list[LoRAAdapter]:
        redis: Redis = await get_redis_client()
        names: set[str] = await cast(
            Awaitable[set[str]], redis.smembers(_REGISTRY_INDEX)
        )
        adapters = []
        for name in names:
            adapter = await self.get_adapter(name)
            if adapter:
                adapters.append(adapter)
        return adapters


@lru_cache
def get_lora_registry() -> LoRARegistryService:
    return LoRARegistryService()
