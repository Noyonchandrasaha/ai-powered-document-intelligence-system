from datetime import UTC, datetime

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", summary="Liveness probe")
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe")
async def readiness() -> dict[str, str]:
    # Add external dependency checks here as the service grows.
    return {"status": "ready", "timestamp": datetime.now(UTC).isoformat()}
