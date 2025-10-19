import pytest
from gateway.services.inference_service import InferenceService
from gateway.clients.triton_client import TritonClient
from gateway.utils.exceptions import InvalidInputError, TritonConnectionError


@pytest.mark.asyncio
async def test_run_inference_success(monkeypatch):
    async def mock_infer(self, model_name, inputs, output_names):
        print("[MOCK] infer called")
        assert model_name == "bert_ensemble"
        assert isinstance(inputs, list)
        assert isinstance(output_names, list)
        return {"outputs": {"bert_emb": [[0.1, 0.2, 0.3]]}}

    monkeypatch.setattr(TritonClient, "infer", mock_infer)

    mock_client = TritonClient()
    service = InferenceService(mock_client)

    result = await service.run_inference(
        "bert_ensemble", [{"name": "TEXTS"}], ["output_names"]
    )

    assert "outputs" in result
    assert "bert_emb" in result["outputs"]
    assert result["outputs"]["bert_emb"][0] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_run_inference_empty_input():
    mock_client = TritonClient()
    service = InferenceService(mock_client)

    with pytest.raises(InvalidInputError):
        await service.run_inference("bert_ensemble", [], ["output_names"])


@pytest.mark.asyncio
async def test_run_inference_failure(monkeypatch):
    async def mock_infer(self, model_name, inputs, output_names):
        raise RuntimeError("mock failure")

    monkeypatch.setattr(TritonClient, "infer", mock_infer)

    service = InferenceService(TritonClient())

    with pytest.raises(TritonConnectionError) as e:
        await service.run_inference(
            "bert_ensemble", [{"name": "TEXTS"}], ["output_names"]
        )

    assert "mock failure" in str(e.value)
