from fastapi.testclient import TestClient
from gateway.main import app


def test_inference_stub():
    headers = {"x-api-key": "test-key"}
    payload = {"model_name": "resnet50", "inputs": {"image": "base64string"}}
    with TestClient(app) as client:
        resp = client.post("/infer", json=payload, headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "class" in body
        assert "confidence" in body
