import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from gateway.middlewares.circuit_breaker.middleware import circuit_breaker_middleware
from gateway.routers import health, inference, version
from gateway.middlewares.request_id import add_request_id
from gateway.middlewares.auth import auth_middleware
from gateway.middlewares.rate_limit import rate_limiter
from gateway.middlewares.metrics import metrics_middleware
from gateway.clients.redis_client import RedisClient, get_redis_client
from gateway.utils.exceptions import register_exception_handlers


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Initializing Redis connection...")
    await get_redis_client()
    yield
    print("[Shutdown] Closing Redis connection...")
    await RedisClient.close()


app = FastAPI(
    title="Triton Inference API Gateway",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(inference.router, prefix="/infer", tags=["Inference"])
app.include_router(version.router, prefix="/version", tags=["Version"])


app.middleware("http")(add_request_id)
app.middleware("http")(auth_middleware)
app.middleware("http")(circuit_breaker_middleware)
app.middleware("http")(rate_limiter)
app.middleware("http")(metrics_middleware)

register_exception_handlers(app)

if __name__ == "__main__":
    uvicorn.run(
        "gateway.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
