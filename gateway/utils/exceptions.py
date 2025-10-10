from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class TritonConnectionError(HTTPException):
    def __init__(self, detail: str = "Failed to connect to Triton Inference Server"):
        super().__init__(status_code=503, detail=detail)


class InvalidInputError(HTTPException):
    def __init__(self, detail: str = "Invalid input format"):
        super().__init__(status_code=400, detail=detail)


async def http_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "path": request.url.path},
        )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Unexpected error",
            "detail": str(exc),
            "path": request.url.path,
        },
    )


async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": request.url.path,
        },
    )
