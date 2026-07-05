import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from gateway.middlewares.circuit_breaker.middleware import _select_breaker
from gateway.middlewares.throttle import _get_endpoint, throttle_middleware


# --- circuit breaker path selection ---


def test_select_breaker_infer():
    assert _select_breaker("/infer") == "triton"
    assert _select_breaker("/infer/") == "triton"


def test_select_breaker_generate():
    # /generate is owned by the router (graceful degradation), so the
    # circuit-breaker middleware skips it rather than short-circuiting to 503.
    assert _select_breaker("/generate") is None
    assert _select_breaker("/generate/stream") is None


def test_select_breaker_other():
    assert _select_breaker("/health") == "global"
    assert _select_breaker("/models") == "global"


# --- throttle endpoint detection ---


def test_get_endpoint_infer():
    assert _get_endpoint("/infer") == "infer"
    assert _get_endpoint("/infer/something") == "infer"


def test_get_endpoint_generate():
    assert _get_endpoint("/generate") == "generate"


def test_get_endpoint_not_throttled():
    assert _get_endpoint("/health") is None
    assert _get_endpoint("/models") is None
    assert _get_endpoint("/version") is None


# --- throttle middleware passthrough ---


@pytest.mark.asyncio
async def test_throttle_skips_non_throttled_paths():
    """Non-throttled paths should pass through without headers."""
    app = FastAPI()
    app.middleware("http")(throttle_middleware)

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok"})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    assert "X-RateLimit-Limit" not in resp.headers


class _FakeRedis:
    def __init__(self, eval_result):
        self._eval_result = eval_result

    async def eval(self, *args, **kwargs):
        return self._eval_result


@pytest.mark.asyncio
async def test_throttle_returns_429_when_rejected(monkeypatch):
    """When the Lua reports a rejection (allowed=0), the middleware must return
    429. Regression guard: the old `current > limit` check never fired, so a
    rejected request silently passed through."""

    async def _fake_get_redis():
        return _FakeRedis([0, 120, 120, 5])  # allowed=0, current, limit, retry_after

    monkeypatch.setattr(
        "gateway.middlewares.throttle.get_redis_client", _fake_get_redis
    )

    app = FastAPI()
    app.middleware("http")(throttle_middleware)

    @app.post("/infer")
    async def infer():
        return JSONResponse({"ok": True})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/infer")

    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "5"
    assert resp.headers["X-RateLimit-Remaining"] == "0"


@pytest.mark.asyncio
async def test_throttle_allows_when_under_limit(monkeypatch):
    async def _fake_get_redis():
        return _FakeRedis([1, 5, 120, 0])  # allowed=1

    monkeypatch.setattr(
        "gateway.middlewares.throttle.get_redis_client", _fake_get_redis
    )

    app = FastAPI()
    app.middleware("http")(throttle_middleware)

    @app.post("/infer")
    async def infer():
        return JSONResponse({"ok": True})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/infer")

    assert resp.status_code == 200
    assert resp.headers["X-RateLimit-Remaining"] == "115"  # 120 - 5
