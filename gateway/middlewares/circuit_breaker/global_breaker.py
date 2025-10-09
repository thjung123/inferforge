from .base import CircuitBreaker

global_breaker = CircuitBreaker(name="global", failure_threshold=5, recovery_time=3)
