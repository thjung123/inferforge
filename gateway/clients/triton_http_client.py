import httpx
from gateway.config import get_settings
from gateway.utils.exceptions import ModelNotFoundError
from gateway.utils.logger import gateway_logger as logger


class TritonHttpClient:

    def __init__(self):
        settings = get_settings()
        self._base_url = settings.triton_http_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=60.0)

    async def get_model_index(self) -> list[dict]:
        resp = await self._client.post("/v2/repository/index", json={})
        resp.raise_for_status()
        return resp.json()

    async def load_model(self, model_name: str) -> None:
        resp = await self._client.post(f"/v2/repository/models/{model_name}/load")
        if resp.status_code != 200:
            detail = resp.text
            raise ModelNotFoundError(
                detail=f"Triton load failed for {model_name}: {detail}"
            )
        logger.info(f"[TritonHTTP] Loaded model: {model_name}")

    async def unload_model(self, model_name: str) -> None:
        resp = await self._client.post(f"/v2/repository/models/{model_name}/unload")
        if resp.status_code != 200:
            detail = resp.text
            raise ModelNotFoundError(
                detail=f"Triton unload failed for {model_name}: {detail}"
            )
        logger.info(f"[TritonHTTP] Unloaded model: {model_name}")

    async def close(self):
        await self._client.aclose()
        logger.info("[TritonHTTP] Connection closed")
