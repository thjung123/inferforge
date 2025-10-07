from functools import lru_cache


class TritonClient:
    def infer(self, model_name: str, inputs: dict) -> dict:
        return {"class": "cat", "confidence": 0.97}


@lru_cache
def get_triton_client() -> TritonClient:
    return TritonClient()
