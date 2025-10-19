from abc import ABC, abstractmethod
from gateway.services.inference_service import InferenceService


class BaseManager(ABC):
    MODEL_NAME: str

    def __init__(self, service: InferenceService):
        self.service = service

    @abstractmethod
    async def run(self, *args, **kwargs):
        pass
