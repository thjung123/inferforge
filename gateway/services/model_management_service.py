from functools import lru_cache

from gateway.clients.builder_client import get_builder_client
from gateway.clients.triton_http_client import get_triton_http_client
from gateway.utils.logger import gateway_logger as logger


class ModelManagementService:
    def __init__(self):
        self._builder = get_builder_client()
        self._triton = get_triton_http_client()

    async def register(
        self, model_type: str, instance_count: int | None = None
    ) -> dict:
        result = await self._builder.build(model_type, instance_count)
        logger.info(
            f"[ModelMgmt] Build submitted: job_id={result['job_id']} model={result['model_name']}"
        )
        return result

    async def list_models(self) -> list[dict]:
        index = await self._triton.get_model_index()
        return index

    async def load_model(self, name: str) -> None:
        await self._triton.load_model(name)

    async def unload_model(self, name: str) -> None:
        await self._triton.unload_model(name)


@lru_cache
def get_model_management_service() -> ModelManagementService:
    return ModelManagementService()
