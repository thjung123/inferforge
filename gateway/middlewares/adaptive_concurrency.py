from collections import deque
from functools import lru_cache

from gateway.config import get_settings
from gateway.utils.logger import gateway_logger as logger


class AdaptiveConcurrencyLimiter:
    """Dynamically adjusts concurrency limit based on response latency.

    When latency is low, concurrency limit increases (GPU has headroom).
    When latency is high, concurrency limit decreases (GPU is saturating).
    """

    def __init__(
        self,
        *,
        initial_limit: int,
        min_limit: int = 2,
        max_limit: int = 64,
        target_latency: float = 2.0,
        window_size: int = 20,
        increase_step: int = 2,
        decrease_factor: float = 0.75,
    ):
        self._min_limit = min_limit
        self._max_limit = max_limit
        self._target_latency = target_latency
        self._window: deque[float] = deque(maxlen=window_size)
        self._increase_step = increase_step
        self._decrease_factor = decrease_factor

        self._current_limit = initial_limit
        self._in_flight = 0

    @property
    def current_limit(self) -> int:
        return self._current_limit

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def avg_latency(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    def is_available(self) -> bool:
        return self._in_flight < self._current_limit

    async def acquire(self) -> bool:
        if not self.is_available():
            return False
        self._in_flight += 1
        return True

    def release(self, latency: float) -> None:
        self._in_flight -= 1
        self._window.append(latency)
        self._adjust()

    def _adjust(self) -> None:
        if len(self._window) < (self._window.maxlen or 0) // 2:
            return

        avg = self.avg_latency
        old_limit = self._current_limit

        if avg < self._target_latency * 0.7:
            new_limit = min(self._current_limit + self._increase_step, self._max_limit)
        elif avg > self._target_latency:
            new_limit = max(
                int(self._current_limit * self._decrease_factor), self._min_limit
            )
        else:
            return

        if new_limit != old_limit:
            self._current_limit = new_limit
            logger.info(
                f"[AdaptiveConcurrency] limit {old_limit} → {new_limit} "
                f"(avg_latency={avg:.3f}s, target={self._target_latency}s)"
            )


@lru_cache
def get_primary_limiter() -> AdaptiveConcurrencyLimiter:
    settings = get_settings()
    return AdaptiveConcurrencyLimiter(
        initial_limit=settings.concurrency_limit_generate_primary,
        max_limit=settings.concurrency_limit_generate_primary * 2,
        target_latency=2.0,
    )


@lru_cache
def get_fallback_limiter() -> AdaptiveConcurrencyLimiter:
    settings = get_settings()
    return AdaptiveConcurrencyLimiter(
        initial_limit=settings.concurrency_limit_generate_fallback,
        max_limit=settings.concurrency_limit_generate_fallback * 2,
        target_latency=1.0,
    )
