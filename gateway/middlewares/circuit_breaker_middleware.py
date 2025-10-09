from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
from gateway.utils.logger import logger
from .circuit_breaker.manager import breaker_manager


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        global_breaker = breaker_manager.get("global")
        if not global_breaker.allow_request():
            return Response("Global circuit open", status_code=503)
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            global_breaker.record_failure()
            logger.error(f"[CircuitBreaker] Global failure: {e}")
            return Response("Internal error", status_code=500)
