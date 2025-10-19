from gateway.services.inference_manager.bert_manager import BertManager
from gateway.services.inference_manager.clip_manager import ClipManager
from gateway.services.inference_service import InferenceService


class InferenceDispatcher:
    def __init__(self, service: InferenceService):
        self.service = service
        self.managers = {
            "bert_ensemble": BertManager(service),
            "clip_ensemble": ClipManager(service),
        }

    async def run(self, model_name: str, inputs: dict):
        manager = self.managers.get(model_name)
        if not manager:
            raise ValueError(f"Unknown model: {model_name}")
        return await manager.run(**inputs)
