from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import HTTPException

from gateway.utils.logger import gateway_logger as logger
from gateway.middlewares.request_id import request_id_ctx


class TritonConnectionError(HTTPException):
    def __init__(self, detail: str = "Failed to connect to Triton Inference Server"):
        super().__init__(status_code=503, detail=detail)


class InvalidInputError(HTTPException):
    def __init__(self, detail: str = "Invalid input format"):
        super().__init__(status_code=400, detail=detail)


async def http_exception_handler(request: Request, exc: Exception):
    request_id = request_id_ctx.get()
    if isinstance(exc, HTTPException):
        logger.warning(
            f"[HTTPException] {exc.status_code} {exc.detail} | path={request.url.path}",
            extra={"request_id": request_id},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail or "HTTP error",
                "status_code": exc.status_code,
                "request_id": str(request_id),
            },
        )

    logger.error(
        f"[UnexpectedException] {exc} | path={request.url.path}",
        exc_info=True,
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "status_code": 500,
            "request_id": str(request_id),
        },
    )


async def generic_exception_handler(request: Request, exc: Exception):
    request_id = request_id_ctx.get()
    logger.error(
        f"[UnhandledException] {exc} | path={request.url.path}",
        exc_info=True,
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "status_code": 500,
            "request_id": str(request_id),
        },
    )


def register_exception_handlers(app):
    handlers = [
        (StarletteHTTPException, http_exception_handler),
        (RequestValidationError, http_exception_handler),
        (TritonConnectionError, http_exception_handler),
        (InvalidInputError, http_exception_handler),
        (Exception, generic_exception_handler),
    ]
    for exc_class, handler in handlers:
        app.add_exception_handler(exc_class, handler)
