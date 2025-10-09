from .global_breaker import global_breaker
from .triton_breaker import triton_breaker
from .redis_breaker import redis_breaker


class BreakerManager:
    def __init__(self):
        self.breakers = {
            "global": global_breaker,
            "triton": triton_breaker,
            "redis": redis_breaker,
        }

    def get(self, name: str):
        return self.breakers.get(name)


breaker_manager = BreakerManager()
