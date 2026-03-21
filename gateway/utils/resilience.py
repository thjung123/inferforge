import asyncio
from typing import TypeVar, Callable, Awaitable

T = TypeVar("T")


async def async_retry(
    fn: Callable[..., Awaitable[T]],
    *args,
    max_retries: int = 3,
    base_delay: float = 0.3,
    on_retry: Callable[[int, Exception], None] | None = None,
    **kwargs,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if on_retry:
                on_retry(attempt, e)
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
    raise last_exc  # type: ignore[misc]
