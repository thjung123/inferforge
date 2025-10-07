from fastapi import FastAPI
from gateway.routers import health, inference
from gateway.middlewares.request_id import add_request_id

app = FastAPI(title="Triton Inference API Gateway")

app.middleware("http")(add_request_id)
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(inference.router, prefix="/infer", tags=["Inference"])
