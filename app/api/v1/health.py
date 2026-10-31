from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, status

from app.core.database import check_database_connection

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", summary="Liveness probe")
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe")
async def readiness() -> dict[str, str]:
    if not await check_database_connection():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not ready.",
        )

    return {"status": "ready", "timestamp": datetime.now(UTC).isoformat()}
