from fastapi import APIRouter, Request, HTTPException

router = APIRouter()


@router.get("/")
async def health_check():
    return {"status": "ok"}


@router.get("/unstable")
async def unstable_endpoint(request: Request):
    app = request.app
    if not hasattr(app.state, "fail_counter"):
        app.state.fail_counter = 0

    app.state.fail_counter += 1
    if app.state.fail_counter < 3:
        raise HTTPException(status_code=500, detail="Simulated temporary failure")
    return {"status": "ok_after_retry"}


@router.get("/fail")
async def fail_endpoint():
    raise HTTPException(
        status_code=500, detail="Always fails (for circuit breaker test)"
    )


@router.get("/reset")
async def reset_endpoint(request: Request):
    request.app.state.fail_counter = 0
    return {"status": "reset"}
