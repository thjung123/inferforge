from fastapi import Request, Response
from gateway.middlewares.circuit_breaker.manager import breaker_manager
from gateway.utils.logger import gateway_logger as logger


async def circuit_breaker_middleware(request: Request, call_next):
    breaker = breaker_manager.get("global")
    if not breaker.allow_request():
        logger.warning("[CircuitBreaker] Request blocked - circuit OPEN")
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
        logger.error(f"[CircuitBreaker] Failure: {e}")
        return Response("Internal server error", status_code=500)
