import logging

import httpx

from builder.config import get_builder_settings

logger = logging.getLogger("builder")

_LOAD_TIMEOUT = 120.0


async def load_model(model_name: str) -> None:
    settings = get_builder_settings()
    base_url = settings.triton_http_url.rstrip("/")
    url = f"{base_url}/v2/repository/models/{model_name}/load"

    async with httpx.AsyncClient(timeout=_LOAD_TIMEOUT) as client:
        resp = await client.post(url)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Triton load failed for {model_name} "
            f"(status={resp.status_code}): {resp.text}"
        )
    logger.info(f"[Deploy] Triton loaded model: {model_name}")
