import pytest
from gateway.clients.triton_client import TritonClient


@pytest.mark.asyncio
async def test_triton_infer_stub(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    async def mock_infer(self, model_name, inputs):
        assert model_name == "resnet50"
        assert isinstance(inputs, dict)
        return {"class": "mock-cat", "confidence": 0.87}

    monkeypatch.setattr(TritonClient, "infer", mock_infer)

    client = TritonClient()
    result = await client.infer("resnet50", {"image": "abc"})

    assert result["class"] == "mock-cat"
    assert 0 <= result["confidence"] <= 1
