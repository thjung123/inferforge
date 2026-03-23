from .base import CircuitBreaker


class BreakerManager:
    def __init__(self):
        self.breakers: dict[str, CircuitBreaker] = {}

    def register(
        self, name: str, failure_threshold: int = 5, recovery_time: int = 30
    ) -> CircuitBreaker:
        breaker = CircuitBreaker(
            name=name, failure_threshold=failure_threshold, recovery_time=recovery_time
        )
        self.breakers[name] = breaker
        return breaker

    def get(self, name: str) -> CircuitBreaker:
        breaker = self.breakers.get(name)
        if breaker is None:
            raise KeyError(f"CircuitBreaker '{name}' not registered")
        return breaker


breaker_manager = BreakerManager()
breaker_manager.register("global", failure_threshold=5, recovery_time=3)
breaker_manager.register("triton", failure_threshold=3, recovery_time=15)
breaker_manager.register("redis", failure_threshold=5, recovery_time=10)
breaker_manager.register("vllm", failure_threshold=3, recovery_time=15)
