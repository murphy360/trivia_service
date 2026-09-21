from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.auth import require_api_key
from app.core.db import get_session
from app.models.job import GenerationJob
from app.pipeline.orchestrator import run_generation_job
from app.schemas.generation import GenerateRequest, JobResponse

router = APIRouter(tags=["generate"], dependencies=[Depends(require_api_key)])


@router.post("/generate", response_model=JobResponse, status_code=202)
async def generate(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> JobResponse:
    job = GenerationJob(params=request.model_dump(mode="json"))
    session.add(job)
    await session.commit()
    await session.refresh(job)

    background_tasks.add_task(run_generation_job, job.id, request)
    return JobResponse.model_validate(job)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: int, session: AsyncSession = Depends(get_session)) -> JobResponse:
    job = await session.get(GenerationJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse.model_validate(job)
