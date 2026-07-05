from fastapi import APIRouter, HTTPException, Request

from gateway.clients.redis_client import get_redis_client
from gateway.config import get_settings

router = APIRouter(redirect_slashes=False)


@router.get("")
@router.get("/")
async def health_check():
    """Liveness."""
    return {"status": "ok"}


@router.get("/ready")
async def readiness_check():
    """Readiness — requires Redis reachable."""
    try:
        redis = await get_redis_client()
        await redis.ping()
    except Exception:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    return {"status": "ready"}


def _require_fault_injection() -> None:
    if not get_settings().enable_fault_injection:
        raise HTTPException(status_code=404, detail="Not found")


@router.get("/unstable")
async def unstable_endpoint(request: Request):
    _require_fault_injection()
    app = request.app
    if not hasattr(app.state, "fail_counter"):
        app.state.fail_counter = 0

    app.state.fail_counter += 1
    if app.state.fail_counter < 3:
        raise HTTPException(status_code=500, detail="Simulated temporary failure")
    return {"status": "ok_after_retry"}


@router.get("/fail")
async def fail_endpoint():
    _require_fault_injection()
    raise HTTPException(
        status_code=500, detail="Always fails (for circuit breaker test)"
    )


@router.get("/reset")
async def reset_endpoint(request: Request):
    _require_fault_injection()
    request.app.state.fail_counter = 0
    return {"status": "reset"}
