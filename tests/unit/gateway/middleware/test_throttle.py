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
    assert _select_breaker("/generate") == "vllm"
    assert _select_breaker("/generate/stream") == "vllm"


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
