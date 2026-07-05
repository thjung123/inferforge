from fastapi.testclient import TestClient
from gateway.main import app

client = TestClient(app)


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readiness_ok_when_redis_reachable():
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_fault_injection_hooks_hidden_by_default():
    # enable_fault_injection defaults False → anonymous self-DoS surface is gone
    assert client.get("/health/fail").status_code == 404
    assert client.get("/health/unstable").status_code == 404
