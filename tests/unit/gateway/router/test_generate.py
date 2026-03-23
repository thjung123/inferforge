from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from gateway.clients.vllm_client import VLLMClient, get_vllm_fallback, get_vllm_primary
from gateway.routers.generate import router

MOCK_RESULT = {
    "choices": [{"message": {"content": "Hello!"}}],
    "model": "test-model",
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}

FALLBACK_RESULT = {
    "choices": [{"message": {"content": "Fallback response"}}],
    "model": "fallback-model",
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}


def _make_mock_client(result: dict) -> VLLMClient:
    mock = AsyncMock(spec=VLLMClient)
    mock.generate = AsyncMock(return_value=result)
    return mock


def _make_failing_client() -> VLLMClient:
    mock = AsyncMock(spec=VLLMClient)
    mock.generate = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
    return mock


@pytest.fixture
def mock_app() -> Any:
    app = FastAPI()
    app.include_router(router, prefix="/generate")
    return app


@pytest.mark.asyncio
async def test_generate_returns_response(mock_app):
    mock_client = _make_mock_client(MOCK_RESULT)
    mock_app.dependency_overrides[get_vllm_primary] = lambda: mock_client
    mock_app.dependency_overrides[get_vllm_fallback] = lambda: mock_client

    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/generate",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "Hello!"
    assert data["model"] == "test-model"
    mock_app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_generate_fallback_on_primary_failure(mock_app):
    mock_app.dependency_overrides[get_vllm_primary] = _make_failing_client
    mock_app.dependency_overrides[get_vllm_fallback] = lambda: _make_mock_client(
        FALLBACK_RESULT
    )

    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/generate",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "Fallback response"
    mock_app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_generate_503_when_all_at_capacity(mock_app):
    """Both primary and fallback limiters full → 503."""
    from unittest.mock import patch

    from gateway.middlewares.adaptive_concurrency import AdaptiveConcurrencyLimiter

    mock_app.dependency_overrides[get_vllm_primary] = _make_failing_client
    mock_app.dependency_overrides[get_vllm_fallback] = lambda: _make_mock_client(
        FALLBACK_RESULT
    )

    # Create a limiter that is always full
    full_limiter = AdaptiveConcurrencyLimiter(initial_limit=1)
    await full_limiter.acquire()  # fill it

    with (
        patch(
            "gateway.routers.generate.get_primary_limiter", return_value=full_limiter
        ),
        patch(
            "gateway.routers.generate.get_fallback_limiter", return_value=full_limiter
        ),
    ):
        transport = ASGITransport(app=mock_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/generate",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )

    assert resp.status_code == 503
    assert "all models at capacity" in resp.json()["error"]
    mock_app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_generate_circuit_open_routes_to_fallback(mock_app):
    """When vLLM circuit breaker is open, request goes directly to fallback."""
    from gateway.middlewares.circuit_breaker.manager import breaker_manager

    mock_app.dependency_overrides[get_vllm_primary] = lambda: _make_mock_client(
        MOCK_RESULT
    )
    mock_app.dependency_overrides[get_vllm_fallback] = lambda: _make_mock_client(
        FALLBACK_RESULT
    )

    breaker = breaker_manager.get("vllm")
    original_open = breaker.open
    original_count = breaker.fail_count
    breaker.open = True
    breaker.fail_count = 99

    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/generate",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "Fallback response"

    breaker.open = original_open
    breaker.fail_count = original_count
    mock_app.dependency_overrides.clear()
