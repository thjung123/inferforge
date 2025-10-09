import time
from gateway.middlewares.circuit_breaker.redis_breaker import redis_breaker


def test_redis_breaker_opens_after_failures():
    cb = redis_breaker
    cb.fail_count = 0
    cb.open = False

    for _ in range(cb.failure_threshold):
        cb.record_failure()

    assert cb.open is True, "RedisBreaker should open after threshold failures"


def test_redis_breaker_recovers_after_timeout():
    cb = redis_breaker
    cb.fail_count = 0
    cb.open = False
    cb.failure_threshold = 1
    cb.recovery_time = 1

    cb.record_failure()
    assert cb.allow_request() is False

    time.sleep(1.1)
    assert cb.allow_request() is True
    assert cb.open is False
    assert cb.fail_count == 0
