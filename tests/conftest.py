import sys
import pathlib

import pytest
import os

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

os.environ.setdefault("API_KEY_WHITELIST", '["test-key"]')


class FakeRedis:
    def __init__(self):
        self.counter = 0
        self.expire_calls = {}
        self._hash_store: dict[str, dict[str, str]] = {}
        self._set_store: dict[str, set[str]] = {}
        self._kv_store: dict[str, str] = {}

    async def incr(self, key):
        self.counter += 1
        return self.counter

    async def expire(self, key, ttl):
        self.expire_calls[key] = ttl

    async def eval(self, script, num_keys, key, *args):
        self.counter += 1
        return str(self.counter)

    async def hset(self, key, mapping=None, **kwargs):
        if key not in self._hash_store:
            self._hash_store[key] = {}
        if mapping:
            self._hash_store[key].update(mapping)
        self._hash_store[key].update(kwargs)
        return len(mapping or kwargs)

    async def hgetall(self, key):
        return dict(self._hash_store.get(key, {}))

    async def sadd(self, key, *values):
        if key not in self._set_store:
            self._set_store[key] = set()
        self._set_store[key].update(values)
        return len(values)

    async def smembers(self, key):
        return set(self._set_store.get(key, set()))

    async def srem(self, key, *values):
        if key in self._set_store:
            self._set_store[key] -= set(values)
        return len(values)

    async def delete(self, key):
        deleted = 1 if key in self._hash_store else 0
        self._hash_store.pop(key, None)
        return deleted

    async def get(self, key):
        return self._kv_store.get(key)

    async def set(self, key, value, ex=None):
        self._kv_store[key] = value
        return True

    async def close(self):
        pass


@pytest.fixture(autouse=True)
def mock_redis(monkeypatch):
    fake_redis = FakeRedis()

    async def fake_get_instance():
        return fake_redis

    async def fake_close():
        pass

    monkeypatch.setattr(
        "gateway.clients.redis_client.RedisClient.get_instance", fake_get_instance
    )
    monkeypatch.setattr("gateway.clients.redis_client.RedisClient.close", fake_close)
