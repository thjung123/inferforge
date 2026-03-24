"""LoRA sync sidecar — polls Redis registry, downloads adapters from MinIO,
and loads/unloads them in the local vLLM instance.

Runs as a standalone process alongside each vLLM pod.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import httpx
from minio import Minio
from redis.asyncio import Redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lora-sync")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "lora-adapters")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8100")
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", "/adapters"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))

_REGISTRY_PREFIX = "lora:adapter:"
_REGISTRY_INDEX = "lora:adapters"


async def _get_registered_adapters(redis: Redis) -> dict[str, dict[str, str]]:
    names: set[str] = await redis.smembers(_REGISTRY_INDEX)  # type: ignore[assignment]
    adapters = {}
    for name in names:
        data: dict[str, str] = await redis.hgetall(f"{_REGISTRY_PREFIX}{name}")  # type: ignore[assignment]
        if data and data.get("status") == "active":
            adapters[name] = data
    return adapters


def _get_local_state() -> dict[str, int]:
    """Read local state file to know which adapters/versions are loaded."""
    state_file = ADAPTER_DIR / ".sync_state.json"
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {}


def _save_local_state(state: dict[str, int]) -> None:
    state_file = ADAPTER_DIR / ".sync_state.json"
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state))


def _download_adapter(minio_client: Minio, s3_path: str, name: str) -> Path:
    local_dir = ADAPTER_DIR / name
    local_dir.mkdir(parents=True, exist_ok=True)

    objects = minio_client.list_objects(MINIO_BUCKET, prefix=s3_path, recursive=True)
    for obj in objects:
        if obj.object_name is None:
            continue
        rel = obj.object_name[len(s3_path) :].lstrip("/")
        if not rel:
            continue
        local_file = local_dir / rel
        local_file.parent.mkdir(parents=True, exist_ok=True)
        minio_client.fget_object(MINIO_BUCKET, obj.object_name, str(local_file))

    logger.info(f"[Sync] Downloaded {name} from {s3_path} → {local_dir}")
    return local_dir


async def _load_lora(name: str, local_path: Path) -> bool:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{VLLM_URL}/v1/load_lora_adapter",
            json={"lora_name": name, "lora_path": str(local_path)},
        )
    if resp.status_code == 200:
        logger.info(f"[Sync] Loaded LoRA adapter: {name}")
        return True
    logger.error(f"[Sync] Failed to load {name}: {resp.text}")
    return False


async def _unload_lora(name: str) -> bool:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{VLLM_URL}/v1/unload_lora_adapter",
            json={"lora_name": name},
        )
    if resp.status_code == 200:
        logger.info(f"[Sync] Unloaded LoRA adapter: {name}")
        return True
    logger.error(f"[Sync] Failed to unload {name}: {resp.text}")
    return False


async def sync_once(redis: Redis, minio_client: Minio) -> None:
    registered = await _get_registered_adapters(redis)
    local_state = _get_local_state()

    # Load new or updated adapters
    for name, data in registered.items():
        version = int(data.get("version", "1"))
        local_version = local_state.get(name, 0)

        if version > local_version:
            logger.info(
                f"[Sync] New/updated adapter: {name} v{local_version} → v{version}"
            )
            local_path = await asyncio.to_thread(
                _download_adapter, minio_client, data["s3_path"], name
            )
            if await _load_lora(name, local_path):
                local_state[name] = version

    # Unload removed adapters
    for name in list(local_state.keys()):
        if name not in registered:
            logger.info(f"[Sync] Adapter removed from registry: {name}")
            await _unload_lora(name)
            del local_state[name]

    _save_local_state(local_state)


async def run() -> None:
    logger.info(
        f"[Sync] Starting LoRA sync sidecar "
        f"(poll={POLL_INTERVAL}s, vllm={VLLM_URL})"
    )

    redis = Redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    # Ensure bucket exists
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
        logger.info(f"[Sync] Created MinIO bucket: {MINIO_BUCKET}")

    try:
        while True:
            try:
                await sync_once(redis, minio_client)
            except Exception as e:
                logger.error(f"[Sync] Error during sync: {e}")
            await asyncio.sleep(POLL_INTERVAL)
    finally:
        await redis.close()


if __name__ == "__main__":
    asyncio.run(run())
