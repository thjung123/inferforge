import pytest

from gateway.services import lora_registry
from gateway.services.lora_registry import LoRAAdapter, LoRARegistryService


class _FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.sets = {}
        self.published = []

    async def hset(self, key, mapping=None, **kw):
        self.hashes.setdefault(key, {}).update(mapping or {})
        return 1

    async def sadd(self, key, *values):
        self.sets.setdefault(key, set()).update(values)
        return len(values)

    async def srem(self, key, *values):
        self.sets.get(key, set()).difference_update(values)
        return len(values)

    async def delete(self, key):
        return 1 if self.hashes.pop(key, None) is not None else 0

    async def publish(self, channel, message):
        import json

        self.published.append((channel, json.loads(message)))
        return 1


@pytest.mark.asyncio
async def test_register_publishes_upsert_event(monkeypatch):
    fake = _FakeRedis()

    async def _get_redis():
        return fake

    monkeypatch.setattr(lora_registry, "get_redis_client", _get_redis)

    await LoRARegistryService().register_adapter(
        LoRAAdapter(name="ko-chat", base_model="m", s3_path="p", version=3)
    )

    assert (
        "lora:events",
        {"action": "upsert", "name": "ko-chat", "version": 3},
    ) in fake.published


@pytest.mark.asyncio
async def test_remove_publishes_remove_event(monkeypatch):
    fake = _FakeRedis()
    fake.hashes["lora:adapter:ko-chat"] = {"name": "ko-chat"}

    async def _get_redis():
        return fake

    monkeypatch.setattr(lora_registry, "get_redis_client", _get_redis)

    ok = await LoRARegistryService().remove_adapter("ko-chat")

    assert ok
    assert ("lora:events", {"action": "remove", "name": "ko-chat"}) in fake.published
