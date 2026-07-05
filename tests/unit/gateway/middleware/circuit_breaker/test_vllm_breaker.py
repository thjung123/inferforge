import time

from gateway.middlewares.circuit_breaker.manager import breaker_manager


def test_vllm_breaker_registered():
    breaker = breaker_manager.get("vllm")
    assert breaker.name == "vllm"
    assert breaker.failure_threshold == 3
    assert breaker.recovery_time == 15


def test_vllm_breaker_opens_after_failures():
    breaker = breaker_manager.get("vllm")
    breaker.fail_count = 0
    breaker.open = False

    for _ in range(3):
        breaker.record_failure()

    assert breaker.open is True
    assert breaker.allow_request() is False


def test_vllm_breaker_recovers_after_timeout():
    breaker = breaker_manager.get("vllm")
    breaker.fail_count = 3
    breaker.open = True
    breaker.half_open = False
    breaker.last_failure = time.time() - 20  # 15s recovery passed

    # one probe allowed (half-open), then fully closed on success
    assert breaker.allow_request() is True
    assert breaker.half_open is True
    assert breaker.allow_request() is False
    breaker.record_success()
    assert breaker.open is False
    assert breaker.half_open is False
