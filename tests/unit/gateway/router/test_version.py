from fastapi.testclient import TestClient
from gateway.main import app

client = TestClient(app)


def test_version_endpoint():
    headers = {"x-api-key": "test-key"}
    resp = client.get("/version", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "version" in body
    assert "commit" in body
    assert "build_time" in body
