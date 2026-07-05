import time
from gateway.middlewares.circuit_breaker.manager import breaker_manager


def test_global_breaker_opens_after_failures():
    cb = breaker_manager.get("global")
    cb.fail_count = 0
    cb.open = False

    for _ in range(cb.failure_threshold):
        cb.record_failure()

    assert cb.open is True, "GlobalBreaker should open after threshold failures"


def test_global_breaker_recovers_after_timeout():
    cb = breaker_manager.get("global")
    cb.fail_count = 0
    cb.open = False
    cb.failure_threshold = 1
    cb.recovery_time = 1

    cb.record_failure()
    assert cb.allow_request() is False

    time.sleep(1.1)
    # recovery window elapsed → one probe allowed (half-open), others still blocked
    assert cb.allow_request() is True
    assert cb.half_open is True
    assert cb.allow_request() is False

    # a successful probe fully closes the breaker
    cb.record_success()
    assert cb.open is False
    assert cb.half_open is False
    assert cb.fail_count == 0
