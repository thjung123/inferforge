import pytest
from gateway.utils.resilience import resilient_call


@pytest.mark.asyncio
async def test_resilient_call_success():
    called = {"count": 0}

    @resilient_call(max_attempts=3)
    async def sample_fn(x):
        called["count"] += 1
        return x * 2

    result = await sample_fn(5)
    assert result == 10
    assert called["count"] == 1


@pytest.mark.asyncio
async def test_resilient_call_retries():
    called = {"count": 0}

    @resilient_call(max_attempts=3)
    async def flaky_fn():
        called["count"] += 1
        if called["count"] < 3:
            raise Exception("fail")
        return "ok"

    result = await flaky_fn()
    assert result == "ok"
    assert called["count"] == 3
