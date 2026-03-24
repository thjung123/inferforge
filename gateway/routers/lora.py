import logging

from fastapi import APIRouter, HTTPException

from gateway.schemas.lora import LoRAAdapterResponse, LoRARegisterRequest
from gateway.services.lora_registry import (
    LoRAAdapter,
    get_adapter,
    list_adapters,
    register_adapter,
    remove_adapter,
)

router = APIRouter()
logger = logging.getLogger("gateway")


@router.post("/register", response_model=LoRAAdapterResponse)
async def register(req: LoRARegisterRequest):
    existing = await get_adapter(req.name)
    version = (existing.version + 1) if existing else 1

    adapter = LoRAAdapter(
        name=req.name,
        base_model=req.base_model,
        s3_path=req.s3_path,
        version=version,
    )
    await register_adapter(adapter)

    return LoRAAdapterResponse(
        name=adapter.name,
        base_model=adapter.base_model,
        s3_path=adapter.s3_path,
        version=adapter.version,
        status=adapter.status,
    )


@router.delete("/{name}")
async def remove(name: str):
    deleted = await remove_adapter(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Adapter '{name}' not found")
    return {"message": f"Adapter '{name}' removed"}


@router.get("", response_model=list[LoRAAdapterResponse])
async def list_all():
    adapters = await list_adapters()
    return [
        LoRAAdapterResponse(
            name=a.name,
            base_model=a.base_model,
            s3_path=a.s3_path,
            version=a.version,
            status=a.status,
        )
        for a in adapters
    ]


@router.get("/{name}", response_model=LoRAAdapterResponse)
async def get(name: str):
    adapter = await get_adapter(name)
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Adapter '{name}' not found")
    return LoRAAdapterResponse(
        name=adapter.name,
        base_model=adapter.base_model,
        s3_path=adapter.s3_path,
        version=adapter.version,
        status=adapter.status,
    )
