from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from gateway.routers import health, inference, version
from gateway.middlewares.request_id import add_request_id
from gateway.utils.metrics import metrics_middleware
from gateway.utils.exceptions import (
    http_exception_handler,
    generic_exception_handler,
    TritonConnectionError,
    InvalidInputError,
)

app = FastAPI(title="Triton Inference API Gateway")

app.middleware("http")(metrics_middleware)
app.middleware("http")(add_request_id)

app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, http_exception_handler)
app.add_exception_handler(TritonConnectionError, http_exception_handler)
app.add_exception_handler(InvalidInputError, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(inference.router, prefix="/infer", tags=["Inference"])
app.include_router(version.router, prefix="/version", tags=["Version"])
