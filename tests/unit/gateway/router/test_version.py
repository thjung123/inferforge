from fastapi.testclient import TestClient
from gateway.main import app

client = TestClient(app)


def test_version_endpoint():
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert "version" in body
    assert "commit" in body
    assert "build_time" in body
