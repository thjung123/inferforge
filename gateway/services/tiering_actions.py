"""Tiering side-effects (Triton load/unload) and the reaper loop. Archive currently
just unloads from Triton; object-store eviction and restore are placeholders."""

import asyncio
import time

import httpx

from gateway.clients.redis_client import get_redis_client
from gateway.config import get_settings
from gateway.services.model_tiering import ModelTieringService
from gateway.utils.logger import gateway_logger as logger


def _repo_url(model: str, action: str) -> str:
    base = get_settings().triton_http_url.rstrip("/")
    return f"{base}/v2/repository/models/{model}/{action}"


async def triton_unload(model: str) -> None:
    async with httpx.AsyncClient(timeout=60.0) as client:
        (await client.post(_repo_url(model, "unload"))).raise_for_status()
    logger.info(f"[Tiering] unloaded {model} from Triton")


async def triton_load(model: str) -> None:
    async with httpx.AsyncClient(timeout=120.0) as client:
        (await client.post(_repo_url(model, "load"))).raise_for_status()
    logger.info(f"[Tiering] loaded {model} into Triton")


async def archive(model: str) -> None:
    await triton_unload(model)


async def restore(model: str) -> None:
    return None


async def run_reaper(interval: float) -> None:
    """Background loop: periodically demote idle models off the GPU."""
    logger.info(f"[Tiering] reaper started (interval={interval}s)")
    while True:
        await asyncio.sleep(interval)
        try:
            redis = await get_redis_client()
            svc = ModelTieringService(redis)
            models = await svc.known_models()
            await svc.reap(models, time.time(), unload=triton_unload, archive=archive)
        except Exception as e:
            logger.error(f"[Tiering] reaper error: {e}")
