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

    async def incr(self, key):
        self.counter += 1
        return self.counter

    async def expire(self, key, ttl):
        self.expire_calls[key] = ttl

    async def eval(self, script, num_keys, key, window):
        self.counter += 1
        return str(self.counter)

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
