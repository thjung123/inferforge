import pytest
import numpy as np
from fastapi.testclient import TestClient
from gateway.main import app
from gateway.services.inference_service import InferenceService, get_inference_service


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def override_service(monkeypatch):
    async def mock_run_inference(self, model_name, inputs, *args, **kwargs):
        if model_name == "bert_ensemble":
            return {"bert_emb": np.array([[0.1, 0.2, 0.3]], dtype=np.float32)}
        elif model_name == "clip_ensemble":
            return {"similarity": np.array([[0.99]], dtype=np.float32)}
        else:
            raise ValueError("unknown model")

    monkeypatch.setattr(InferenceService, "run_inference", mock_run_inference)

    mock_service = InferenceService(client=None)
    app.dependency_overrides[get_inference_service] = lambda: mock_service

    yield
    app.dependency_overrides.clear()


def test_infer_bert(client, override_service):
    payload = {"model_name": "bert_ensemble", "inputs": {"texts": ["hello world"]}}
    headers = {"x-api-key": "test-key"}

    res = client.post("/infer/", json=payload, headers=headers)
    assert res.status_code == 200

    data = res.json()
    assert "bert_emb" in data
    assert isinstance(data["bert_emb"], list)
    assert len(data["bert_emb"][0]) == 3


def test_infer_clip(client, override_service):
    payload = {
        "model_name": "clip_ensemble",
        "inputs": {
            "image_urls": ["https://example.com/cat.png"],
            "texts": ["a cat"],
        },
    }
    headers = {"x-api-key": "test-key"}

    res = client.post("/infer/", json=payload, headers=headers)
    assert res.status_code == 200

    data = res.json()
    assert "similarity" in data
    assert isinstance(data["similarity"], list)
    assert isinstance(data["similarity"][0], list)
    assert data["similarity"][0][0] == pytest.approx(0.99)
