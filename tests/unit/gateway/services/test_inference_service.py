import pytest
from gateway.services.inference_service import InferenceService
from gateway.clients.triton_client import TritonClient
from gateway.utils.exceptions import InvalidInputError


@pytest.mark.asyncio
async def test_inference_service_success():
    service = InferenceService(TritonClient())
    result = await service.run_inference("dummy_model", {"data": [1, 2, 3]})
    assert "class" in result


@pytest.mark.asyncio
async def test_inference_service_invalid_input():
    service = InferenceService(TritonClient())
    with pytest.raises(InvalidInputError):
        await service.run_inference("dummy_model", {})
