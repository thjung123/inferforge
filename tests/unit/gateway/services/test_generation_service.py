from unittest.mock import AsyncMock

import pytest

from gateway.clients.vllm_client import VLLMClient
from gateway.services.generation_service import GenerationService

MOCK_RESULT = {
    "choices": [{"message": {"content": "Hello!"}}],
    "model": "test-model",
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}


@pytest.mark.asyncio
async def test_generate_calls_client():
    mock_client = AsyncMock(spec=VLLMClient)
    mock_client.generate = AsyncMock(return_value=MOCK_RESULT)
    service = GenerationService(mock_client)

    result = await service.generate(
        model="test",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=64,
        temperature=0.5,
    )

    assert result["choices"][0]["message"]["content"] == "Hello!"
    mock_client.generate.assert_called_once_with(
        model="test",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=64,
        temperature=0.5,
        stream=False,
    )


@pytest.mark.asyncio
async def test_generate_stream_yields_chunks():
    async def fake_stream(*args, **kwargs):
        for chunk in [b"data: chunk1\n", b"data: chunk2\n"]:
            yield chunk

    mock_client = AsyncMock(spec=VLLMClient)
    mock_client.generate_stream = fake_stream
    service = GenerationService(mock_client)

    chunks = []
    async for chunk in service.generate_stream(
        model="test",
        messages=[{"role": "user", "content": "Hi"}],
    ):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0] == b"data: chunk1\n"
