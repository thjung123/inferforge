from fastapi.testclient import TestClient
from gateway.main import app

client = TestClient(app)


def test_inference_stub():
    payload = {"model_name": "resnet50", "inputs": {"image": "base64string"}}
    resp = client.post("/infer", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "class" in body
    assert "confidence" in body
