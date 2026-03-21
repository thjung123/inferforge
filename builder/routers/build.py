import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException
from starlette.status import HTTP_202_ACCEPTED

from builder.schemas import BuildRequest, BuildResponse, JobState, JobStatus
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
    if body.instance_count is not None:
        preset.setdefault("triton", {})["instance_count"] = body.instance_count
    model_name = preset["model_name"]

    job_id = uuid.uuid4().hex[:12]

    await tracker.create(job_id, model_name)

    asyncio.create_task(run_build_pipeline(job_id, preset, tracker))

    return BuildResponse(
        job_id=job_id,
        model_name=model_name,
        status=JobState.PENDING,
    )


@router.get("/build/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    tracker: JobTracker = Depends(get_job_tracker),
):
    data = await tracker.get(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatus(
        job_id=data["job_id"],
        model_name=data["model_name"],
        status=JobState(data["status"]),
        error=data.get("error") or None,
    )


@router.get("/build", response_model=list[JobStatus])
async def list_jobs(
    tracker: JobTracker = Depends(get_job_tracker),
):
    jobs = await tracker.list_all()
    return [
        JobStatus(
            job_id=d["job_id"],
            model_name=d["model_name"],
            status=JobState(d["status"]),
            error=d.get("error") or None,
        )
        for d in jobs
    ]
