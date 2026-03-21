import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from builder.config import get_builder_settings
from builder.routers.build import router as build_router
from builder.services.job_tracker import JobTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("builder")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_builder_settings()
    logger.info(f"[Startup] Connecting to Redis: {settings.redis_url}")
    redis = await Redis.from_url(
        settings.redis_url, encoding="utf-8", decode_responses=True
    )
    JobTracker.initialize(redis)
    yield
    logger.info("[Shutdown] Closing Redis ...")
    await redis.close()


app = FastAPI(title="Model Builder Sidecar", version="1.0.0", lifespan=lifespan)
app.include_router(build_router)
