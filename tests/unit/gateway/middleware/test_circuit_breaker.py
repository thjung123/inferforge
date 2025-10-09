import time
from gateway.middlewares.circuit_breaker import CircuitBreaker


def test_circuit_opens_after_failures():
    cb = CircuitBreaker(failure_threshold=3, recovery_time=10)
    for _ in range(3):
        cb.record_failure()
    assert cb.open is True


def test_circuit_recovers_after_timeout():
    cb = CircuitBreaker(failure_threshold=1, recovery_time=1)
    cb.record_failure()
    assert cb.allow_request() is False
    time.sleep(1.1)
    assert cb.allow_request() is True
    assert cb.open is False
    assert cb.fail_count == 0
