import pytest
import jwt
from fastapi import Request, HTTPException
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
    token = jwt.encode({"user": "test"}, settings.jwt_secret, algorithm="HS256")

    req = make_request({"Authorization": f"Bearer {token}"})
    call_next = DummyCallNext()

    result = await auth_middleware(req, call_next)
    assert result == "ok"
    assert call_next.called


@pytest.mark.asyncio
async def test_invalid_jwt():
    req = make_request({"Authorization": "Bearer invalid"})
    call_next = DummyCallNext()

    with pytest.raises(HTTPException) as e:
        await auth_middleware(req, call_next)
    assert e.value.status_code in (401, 403)
