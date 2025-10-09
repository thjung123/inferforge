import pytest
from gateway.clients.triton_client import TritonClient


@pytest.mark.asyncio
async def test_triton_infer_stub():
    client = TritonClient()
    result = await client.infer("resnet50", {"image": "abc"})
    assert result["class"] == "cat"
    assert 0 <= result["confidence"] <= 1
