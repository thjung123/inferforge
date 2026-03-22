from fastapi import Request, Response
from gateway.middlewares.circuit_breaker.manager import breaker_manager
from gateway.utils.logger import gateway_logger as logger


def _select_breaker(path: str) -> str:
    if path.startswith("/generate"):
        return "vllm"
    if path.startswith("/infer"):
        return "triton"
    return "global"


async def circuit_breaker_middleware(request: Request, call_next):
    breaker_name = _select_breaker(request.url.path)
    breaker = breaker_manager.get(breaker_name)
    if not breaker.allow_request():
        logger.warning(
            f"[CircuitBreaker:{breaker_name}] Request blocked - circuit OPEN"
        )
        return Response("Circuit open", status_code=503)
    try:
        response = await call_next(request)
        if response.status_code >= 500:
            breaker.record_failure()
        else:
            breaker.record_success()
        return response
    except Exception as e:
        breaker.record_failure()
        logger.error(f"[CircuitBreaker:{breaker_name}] Failure: {e}")
        return Response("Internal server error", status_code=500)
