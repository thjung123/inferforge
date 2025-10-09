from .base import CircuitBreaker

triton_breaker = CircuitBreaker(name="triton", failure_threshold=3, recovery_time=20)
