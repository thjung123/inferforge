import time
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import AsyncClient, ASGITransport
from gateway.middlewares.circuit_breaker.manager import breaker_manager
from gateway.middlewares.circuit_breaker_middleware import CircuitBreakerMiddleware


@pytest.fixture
def mock_app():
    app = FastAPI()
    app.add_middleware(CircuitBreakerMiddleware)  # type: ignore[arg-type]

    @app.get("/ok")
    async def ok_route():
        return JSONResponse({"status": "ok"})

    @app.get("/fail")
    async def fail_route():
        raise RuntimeError("Simulated failure")

    return app


@pytest.mark.asyncio
async def test_global_breaker_blocks_after_failures(mock_app):
    transport = ASGITransport(app=mock_app)
    client = AsyncClient(transport=transport, base_url="http://test")
    global_breaker = breaker_manager.get("global")

    global_breaker.fail_count = 0
    global_breaker.open = False
    global_breaker.failure_threshold = 2
    global_breaker.recovery_time = 1

    r1 = await client.get("/ok")
    assert r1.status_code == 200
    assert global_breaker.open is False

    for _ in range(2):
        await client.get("/fail")
    assert global_breaker.open is True

    r2 = await client.get("/ok")
    assert r2.status_code == 503

    time.sleep(1.1)
    r3 = await client.get("/ok")
    assert r3.status_code == 200
    assert global_breaker.open is False
