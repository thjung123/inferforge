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

    route = request.scope.get("route")
    path_label = getattr(route, "path", None) or "unmatched"

    REQUEST_COUNT.labels(
        method=request.method, path=path_label, status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(path=path_label).observe(duration)

    return response
