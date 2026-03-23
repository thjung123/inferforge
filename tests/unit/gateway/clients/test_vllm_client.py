from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from gateway.clients.vllm_client import VLLMClient


@pytest.mark.asyncio
async def test_vllm_client_generate():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello"}}],
        "model": "test",
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(VLLMClient, "__init__", lambda self, url: None):
        client = VLLMClient.__new__(VLLMClient)
        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=mock_response)
        client._base_url = "http://test:8100"

        result = await client.generate(
            model="test",
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert result["choices"][0]["message"]["content"] == "Hello"
    client._client.post.assert_called_once()


@pytest.mark.asyncio
async def test_vllm_client_health_success():
    with patch.object(VLLMClient, "__init__", lambda self, url: None):
        client = VLLMClient.__new__(VLLMClient)
        client._client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        client._client.get = AsyncMock(return_value=mock_resp)

        assert await client.health() is True


@pytest.mark.asyncio
async def test_vllm_client_health_failure():
    with patch.object(VLLMClient, "__init__", lambda self, url: None):
        client = VLLMClient.__new__(VLLMClient)
        client._client = AsyncMock()
        client._client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        assert await client.health() is False
