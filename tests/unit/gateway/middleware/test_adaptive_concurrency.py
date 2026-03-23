import pytest

from gateway.middlewares.adaptive_concurrency import AdaptiveConcurrencyLimiter


def _make_limiter(**kwargs: object) -> AdaptiveConcurrencyLimiter:
    defaults: dict[str, object] = {
        "initial_limit": 8,
        "min_limit": 2,
        "max_limit": 32,
        "target_latency": 2.0,
        "window_size": 10,
        "increase_step": 2,
        "decrease_factor": 0.75,
    }
    defaults.update(kwargs)
    return AdaptiveConcurrencyLimiter(**defaults)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_acquire_and_release():
    limiter = _make_limiter(initial_limit=4)
    assert limiter.is_available()
    assert await limiter.acquire()
    assert limiter.in_flight == 1
    limiter.release(1.0)
    assert limiter.in_flight == 0


@pytest.mark.asyncio
async def test_not_available_when_full():
    limiter = _make_limiter(initial_limit=2)
    await limiter.acquire()
    await limiter.acquire()
    assert not limiter.is_available()


@pytest.mark.asyncio
async def test_limit_increases_on_low_latency():
    limiter = _make_limiter(initial_limit=8, target_latency=2.0, window_size=10)
    initial = limiter.current_limit

    # Fill window with fast responses (well below target * 0.7 = 1.4s)
    for _ in range(10):
        await limiter.acquire()
        limiter.release(0.5)

    assert limiter.current_limit > initial


@pytest.mark.asyncio
async def test_limit_decreases_on_high_latency():
    limiter = _make_limiter(initial_limit=16, target_latency=2.0, window_size=10)
    initial = limiter.current_limit

    # Fill window with slow responses (above target 2.0s)
    for _ in range(10):
        await limiter.acquire()
        limiter.release(3.0)

    assert limiter.current_limit < initial


@pytest.mark.asyncio
async def test_limit_stays_within_bounds():
    limiter = _make_limiter(initial_limit=4, min_limit=2, max_limit=10, window_size=10)

    # Push limit up
    for _ in range(50):
        await limiter.acquire()
        limiter.release(0.1)
    assert limiter.current_limit <= 10

    # Push limit down
    for _ in range(50):
        await limiter.acquire()
        limiter.release(5.0)
    assert limiter.current_limit >= 2


@pytest.mark.asyncio
async def test_avg_latency():
    limiter = _make_limiter(window_size=5)
    assert limiter.avg_latency == 0.0

    for lat in [1.0, 2.0, 3.0]:
        await limiter.acquire()
        limiter.release(lat)

    assert abs(limiter.avg_latency - 2.0) < 0.01
