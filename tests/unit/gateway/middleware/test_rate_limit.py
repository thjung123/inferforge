import json

import pytest
from fastapi import Request
from gateway.middlewares.rate_limit import rate_limiter


@pytest.mark.asyncio
async def test_rate_limit_under_threshold():
    req = Request({"type": "http", "client": ("127.0.0.1", 5000), "headers": []})

    async def call_next(req):
        return "ok"

    result = await rate_limiter(req, call_next)
    assert result == "ok"


@pytest.mark.asyncio
async def test_rate_limit_exceeded(monkeypatch):
    from gateway.clients.redis_client import RedisClient

    fake = await RedisClient.get_instance()
    fake.counter = 101

    req = Request({"type": "http", "client": ("127.0.0.1", 5000), "headers": []})

    async def call_next(req):
        return "ok"

    resp = await rate_limiter(req, call_next)
    assert resp.status_code == 429
    data = json.loads(resp.body)
    assert data["detail"] == "Rate limit exceeded"
