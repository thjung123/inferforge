from datetime import datetime, timedelta, timezone

import pytest
import jwt
from fastapi import Request
from starlette.datastructures import Headers
from gateway.config import get_settings
from gateway.middlewares.auth import auth_middleware


class DummyCallNext:
    def __init__(self):
        self.called = False

    async def __call__(self, req):
        self.called = True
        return "ok"


def make_request(headers: dict | None = None, path: str = "/test") -> Request:
    _headers = Headers(headers or {})
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": _headers.raw,
        "client": ("127.0.0.1", 5000),
        "scheme": "http",
        "server": ("testserver", 80),
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_api_key_whitelist(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "api_key_whitelist", ["key123"])

    req = make_request({"x-api-key": "key123"})
    call_next = DummyCallNext()

    result = await auth_middleware(req, call_next)
    assert result == "ok"
    assert call_next.called


@pytest.mark.asyncio
async def test_valid_jwt(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "jwt_secret", "unit-test-secret")
    token = jwt.encode(
        {"user": "test", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
        settings.jwt_secret,
        algorithm="HS256",
    )

    req = make_request({"Authorization": f"Bearer {token}"})
    call_next = DummyCallNext()

    result = await auth_middleware(req, call_next)
    assert result == "ok"
    assert call_next.called


@pytest.mark.asyncio
async def test_expired_jwt(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "jwt_secret", "unit-test-secret")
    token = jwt.encode(
        {"user": "test", "exp": datetime.now(timezone.utc) - timedelta(hours=1)},
        settings.jwt_secret,
        algorithm="HS256",
    )

    req = make_request({"Authorization": f"Bearer {token}"})
    call_next = DummyCallNext()

    resp = await auth_middleware(req, call_next)
    assert resp.status_code == 401
    assert not call_next.called


@pytest.mark.asyncio
async def test_invalid_jwt(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "jwt_secret", "unit-test-secret")

    req = make_request({"Authorization": "Bearer invalid"})
    call_next = DummyCallNext()

    resp = await auth_middleware(req, call_next)
    # auth returns a proper response (not a raised HTTPException that a wrapping
    # middleware could turn into a 500)
    assert resp.status_code == 401
    assert not call_next.called


@pytest.mark.asyncio
async def test_unauthorized_returns_403(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "api_key_whitelist", [])

    req = make_request({})
    call_next = DummyCallNext()

    resp = await auth_middleware(req, call_next)
    assert resp.status_code == 403
    assert not call_next.called


@pytest.mark.asyncio
async def test_bearer_rejected_when_secret_unset(monkeypatch):
    """Empty jwt_secret disables JWT auth entirely — no forgeable default."""
    settings = get_settings()
    monkeypatch.setattr(settings, "jwt_secret", "")
    token = jwt.encode({"user": "test"}, "anything", algorithm="HS256")

    req = make_request({"Authorization": f"Bearer {token}"})
    call_next = DummyCallNext()

    resp = await auth_middleware(req, call_next)
    assert resp.status_code == 403
    assert not call_next.called
