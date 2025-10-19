import pytest
from fastapi.testclient import TestClient
from gateway.main import app
from gateway.services.inference_service import InferenceService
from gateway.clients.triton_client import get_triton_client


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def override_service(monkeypatch):
    async def mock_run_inference(self, model_name, inputs):
        if model_name == "bert_ensemble":
            return {
                "outputs": [
                    {"name": "bert_emb", "data": [[0.1, 0.2, 0.3]], "shape": [1, 3]}
                ]
            }
        elif model_name == "clip_ensemble":
            return {
                "outputs": [{"name": "similarity", "data": [[0.99]], "shape": [1, 1]}]
            }
        else:
            raise ValueError("unknown model")

    monkeypatch.setattr(InferenceService, "run_inference", mock_run_inference)

    dependency_overrides: dict = getattr(app, "dependency_overrides", {})
    dependency_overrides[get_triton_client] = lambda: None

    yield

    dependency_overrides.clear()


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
