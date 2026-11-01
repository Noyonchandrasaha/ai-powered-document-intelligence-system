from typing import Annotated, Any

from app.core.database import get_database
from app.features.jobs import JobQueue
from fastapi import APIRouter, Depends, HTTPException, Path
from pymongo.asynchronous.database import AsyncDatabase

router = APIRouter(prefix="/jobs", tags=["Jobs"])
JobId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]
Database = Annotated[AsyncDatabase[Any], Depends(get_database)]


@router.get("/{job_id}")
async def job_status(job_id: JobId, database: Database) -> dict[str, Any]:
    job = await JobQueue(database).get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return job


@router.post("/{job_id}/retry", status_code=202)
async def retry_job(job_id: JobId, database: Database) -> dict[str, Any]:
    try:
        job = await JobQueue(database).retry(job_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    if job is None:
        raise HTTPException(404, "Job not found.")
    return job
