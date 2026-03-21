import pytest
from fastapi.testclient import TestClient

from gateway.config import get_settings
from gateway.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def patch_auth(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "api_key_whitelist", ["test-key"])


def test_version_endpoint(client):
    headers = {"x-api-key": "test-key"}
    resp = client.get("/version", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "version" in body
