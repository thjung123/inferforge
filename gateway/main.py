import asyncio
import contextlib
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from gateway.config import get_settings
from gateway.middlewares.circuit_breaker.middleware import circuit_breaker_middleware
from gateway.routers import health, inference, models, version
from gateway.routers.version import APP_VERSION
from gateway.middlewares.request_id import add_request_id
from gateway.middlewares.auth import auth_middleware
from gateway.middlewares.throttle import throttle_middleware
from gateway.middlewares.metrics import metrics_middleware
from gateway.clients.builder_client import get_builder_client
from gateway.clients.redis_client import RedisClient, get_redis_client
from gateway.clients.triton_http_client import get_triton_http_client
from gateway.utils.exceptions import register_exception_handlers
from gateway.utils.logger import gateway_logger as logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Startup] Initializing Redis connection...")
    await get_redis_client()

    reaper_task = None
    settings = get_settings()
    if settings.enable_tiering:
        from gateway.services.tiering_actions import run_reaper

        reaper_task = asyncio.create_task(run_reaper(settings.tiering_reaper_interval))

    yield

    logger.info("[Shutdown] Closing connections...")
    if reaper_task:
        reaper_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reaper_task
    await get_builder_client().close()
    await get_triton_http_client().close()
    await RedisClient.close()


app = FastAPI(
    title="Triton Inference API Gateway",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(inference.router, prefix="/infer", tags=["Inference"])
app.include_router(models.router, prefix="/models", tags=["Models"])
app.include_router(version.router, prefix="/version", tags=["Version"])


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.middleware("http")(circuit_breaker_middleware)
app.middleware("http")(throttle_middleware)
app.middleware("http")(auth_middleware)
app.middleware("http")(metrics_middleware)
app.middleware("http")(add_request_id)

register_exception_handlers(app)

if __name__ == "__main__":
    uvicorn.run(
        "gateway.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )
