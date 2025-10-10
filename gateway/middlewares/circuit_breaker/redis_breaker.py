from .base import CircuitBreaker

redis_breaker = CircuitBreaker(name="redis", failure_threshold=5, recovery_time=10)
