from fastapi import APIRouter, Depends

from gateway.schemas.lora import LoRAAdapterResponse, LoRARegisterRequest
from gateway.services.lora_registry import LoRARegistryService, get_lora_registry
from gateway.utils.exceptions import ModelNotFoundError

router = APIRouter()


@router.post("/register", response_model=LoRAAdapterResponse)
async def register(
    req: LoRARegisterRequest,
    registry: LoRARegistryService = Depends(get_lora_registry),
):
    adapter = await registry.register_adapter_atomic(
        req.name, req.base_model, req.s3_path
    )

    return LoRAAdapterResponse(
        name=adapter.name,
        base_model=adapter.base_model,
        s3_path=adapter.s3_path,
        version=adapter.version,
        status=adapter.status,
    )


@router.delete("/{name}")
async def remove(
    name: str,
    registry: LoRARegistryService = Depends(get_lora_registry),
):
    deleted = await registry.remove_adapter(name)
    if not deleted:
        raise ModelNotFoundError(detail=f"Adapter '{name}' not found")
    return {"message": f"Adapter '{name}' removed"}


@router.get("", response_model=list[LoRAAdapterResponse])
async def list_all(
    registry: LoRARegistryService = Depends(get_lora_registry),
):
    adapters = await registry.list_adapters()
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
async def get(
    name: str,
    registry: LoRARegistryService = Depends(get_lora_registry),
):
    adapter = await registry.get_adapter(name)
    if not adapter:
        raise ModelNotFoundError(detail=f"Adapter '{name}' not found")
    return LoRAAdapterResponse(
        name=adapter.name,
        base_model=adapter.base_model,
        s3_path=adapter.s3_path,
        version=adapter.version,
        status=adapter.status,
    )
