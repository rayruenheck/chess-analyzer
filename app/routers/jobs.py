from enum import Enum

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.services import jobs

router = APIRouter(prefix="/jobs", tags=["jobs"])


class Platform(str, Enum):
    lichess = "lichess"
    chesscom = "chesscom"


class AnalyzeJobRequest(BaseModel):
    platform: Platform
    username: str
    max_games: int = 20


@router.post("/analyze")
async def create_analyze_job(request: AnalyzeJobRequest, background_tasks: BackgroundTasks):
    job_id = await jobs.create_job(request.platform.value, request.username, request.max_games)
    background_tasks.add_task(
        jobs.run_analysis_job, job_id, request.platform.value, request.username, request.max_games
    )
    return {"job_id": job_id, "status": "queued"}


@router.get("/{job_id}")
async def get_job_status(job_id: str):
    job = await jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
