import pytest
import numpy as np
from gateway.clients.triton_client import TritonClient


@pytest.mark.asyncio
async def test_triton_infer_stub(monkeypatch):
    class DummyResponse:
        def get_response(self):
            class Outputs:
                outputs = [type("o", (), {"name": "mock_output"})()]

            return Outputs()

        def as_numpy(self, name):
            return b"mock-cat"

    class DummyClient:
        async def infer(self, *args, **kwargs):
            return DummyResponse()

    async def mock_get_client(self):
        return DummyClient()

    monkeypatch.setattr(TritonClient, "_get_client", mock_get_client)

    client = TritonClient.__new__(TritonClient)

    client.max_retries = 1
    client.base_delay = 0.1
    client.triton_enabled = True
    client.triton_breaker = type(
        "b",
        (),
        {
            "allow_request": lambda *_: True,
            "record_success": lambda *_: None,
            "record_failure": lambda *_: None,
        },
    )()

    result = await client.infer(
        "resnet50",
        [
            {
                "name": "img",
                "shape": [1],
                "datatype": "FP32",
                "data": np.array([0.1], dtype=np.float32),
            }
        ],
        ["mock_output"],
    )

    assert result["mock_output"] == b"mock-cat"
