from fastapi import APIRouter
import os

router = APIRouter(redirect_slashes=False)


@router.get("")
@router.get("/")
async def get_version():
    return {
        "version": os.getenv("APP_VERSION", "0.1.0"),
        "commit": os.getenv("GIT_COMMIT", "unknown"),
        "build_time": os.getenv("BUILD_TIME", "unknown"),
    }
