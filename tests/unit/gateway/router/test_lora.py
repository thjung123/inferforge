from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from gateway.routers.lora import router


@pytest.fixture
def mock_app() -> Any:
    app = FastAPI()
    app.include_router(router, prefix="/lora")
    return app


@pytest.mark.asyncio
async def test_register_adapter(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/lora/register",
            json={
                "name": "ko-chat",
                "base_model": "Qwen/Qwen2.5-7B-Instruct",
                "s3_path": "adapters/ko-chat-lora",
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "ko-chat"
    assert data["version"] == 1
    assert data["status"] == "active"


@pytest.mark.asyncio
async def test_list_adapters(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Register one
        await client.post(
            "/lora/register",
            json={
                "name": "test-adapter",
                "base_model": "test-model",
                "s3_path": "adapters/test",
            },
        )

        resp = await client.get("/lora")

    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_get_adapter(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/lora/register",
            json={
                "name": "get-test",
                "base_model": "test-model",
                "s3_path": "adapters/get-test",
            },
        )

        resp = await client.get("/lora/get-test")

    assert resp.status_code == 200
    assert resp.json()["name"] == "get-test"


@pytest.mark.asyncio
async def test_get_missing_adapter(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/lora/nonexistent")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_remove_adapter(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/lora/register",
            json={
                "name": "remove-test",
                "base_model": "test-model",
                "s3_path": "adapters/remove-test",
            },
        )

        resp = await client.delete("/lora/remove-test")
        assert resp.status_code == 200

        resp = await client.get("/lora/remove-test")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_register_bumps_version(mock_app):
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp1 = await client.post(
            "/lora/register",
            json={
                "name": "versioned",
                "base_model": "test-model",
                "s3_path": "adapters/v1",
            },
        )
        assert resp1.json()["version"] == 1

        resp2 = await client.post(
            "/lora/register",
            json={
                "name": "versioned",
                "base_model": "test-model",
                "s3_path": "adapters/v2",
            },
        )
        assert resp2.json()["version"] == 2
