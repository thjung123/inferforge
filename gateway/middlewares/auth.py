from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.security.utils import get_authorization_scheme_param
import jwt
from jwt import PyJWTError

from gateway.config import get_settings


async def auth_middleware(request: Request, call_next):
    settings = get_settings()

    if request.url.path.startswith(("/health", "/metrics")):
        return await call_next(request)

    api_key = request.headers.get("x-api-key")
    if api_key and api_key in settings.api_key_whitelist:
        return await call_next(request)

    auth_header = request.headers.get("Authorization")
    if auth_header and settings.jwt_secret:
        scheme, token = get_authorization_scheme_param(auth_header)
        if scheme.lower() == "bearer":
            try:
                jwt.decode(
                    token,
                    settings.jwt_secret,
                    algorithms=["HS256"],
                    options={"require": ["exp"]},
                )
                return await call_next(request)
            except PyJWTError:
                return JSONResponse(
                    status_code=401, content={"error": "Invalid JWT token"}
                )

    return JSONResponse(status_code=403, content={"error": "Unauthorized"})
