import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from gateway.middlewares.circuit_breaker.middleware import circuit_breaker_middleware
from gateway.routers import generate, health, inference, lora, models, version
from gateway.middlewares.request_id import add_request_id
from gateway.middlewares.auth import auth_middleware
from gateway.middlewares.throttle import throttle_middleware
from gateway.middlewares.metrics import metrics_middleware
from gateway.clients.builder_client import get_builder_client
from gateway.clients.redis_client import RedisClient, get_redis_client
from gateway.clients.triton_http_client import get_triton_http_client
from gateway.clients.vllm_client import get_vllm_fallback, get_vllm_primary
from gateway.utils.exceptions import register_exception_handlers


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Initializing Redis connection...")
    await get_redis_client()
    yield
    print("[Shutdown] Closing connections...")
    await get_builder_client().close()
    await get_triton_http_client().close()
    await get_vllm_primary().close()
    await get_vllm_fallback().close()
    await RedisClient.close()


app = FastAPI(
    title="Triton Inference API Gateway",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(inference.router, prefix="/infer", tags=["Inference"])
app.include_router(models.router, prefix="/models", tags=["Models"])
app.include_router(generate.router, prefix="/generate", tags=["Generate"])
app.include_router(lora.router, prefix="/lora", tags=["LoRA"])
app.include_router(version.router, prefix="/version", tags=["Version"])


app.middleware("http")(add_request_id)
app.middleware("http")(auth_middleware)
app.middleware("http")(circuit_breaker_middleware)
app.middleware("http")(throttle_middleware)
app.middleware("http")(metrics_middleware)

register_exception_handlers(app)

if __name__ == "__main__":
    uvicorn.run(
        "gateway.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
