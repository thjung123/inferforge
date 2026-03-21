import asyncio
import uuid

from fastapi import APIRouter, Depends
from starlette.status import HTTP_202_ACCEPTED

from builder.schemas import BuildRequest, BuildResponse, JobState
from builder.services.build_pipeline import run_build_pipeline
from builder.services.job_tracker import JobTracker, get_job_tracker
from builder.services.preset_loader import load_preset

router = APIRouter()


@router.post("/build", status_code=HTTP_202_ACCEPTED, response_model=BuildResponse)
async def build_model(
    body: BuildRequest,
    tracker: JobTracker = Depends(get_job_tracker),
):
    preset = load_preset(body.model_type)
    model_name = preset["model_name"]

    job_id = uuid.uuid4().hex[:12]

    await tracker.create(job_id, model_name)

    asyncio.create_task(run_build_pipeline(job_id, preset, tracker))

    return BuildResponse(
        job_id=job_id,
        model_name=model_name,
        status=JobState.PENDING,
    )
