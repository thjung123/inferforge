from gateway.utils.resilience import resilient_call


@resilient_call(max_attempts=3)
async def call_with_retry(fn, *args, **kwargs):
    return await fn(*args, **kwargs)
