from .base import CircuitBreaker

global_breaker = CircuitBreaker(name="global", failure_threshold=10, recovery_time=60)
