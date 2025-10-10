from prometheus_client import Counter, Histogram
from fastapi import Request
from time import time

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "Request latency", ["path"]
)


async def metrics_middleware(request: Request, call_next):
    start = time()
    response = await call_next(request)
    duration = time() - start

    REQUEST_COUNT.labels(
        method=request.method, path=request.url.path, status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(path=request.url.path).observe(duration)

    return response
