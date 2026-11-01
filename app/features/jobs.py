"""Durable MongoDB job queue with atomic claims, expiring leases and retry history."""

from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import uuid4

from app.core.config import get_settings
from app.core.exceptions import AppError
from pymongo import ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError

JobKind = Literal["process", "extract"]


class JobQueue:
    def __init__(self, database: AsyncDatabase[Any]) -> None:
        self.collection = database["jobs"]

    async def ensure_indexes(self) -> None:
        await self.collection.create_index("job_id", unique=True)
        await self.collection.create_index(
            "active_key", unique=True, partialFilterExpression={"active_key": {"$type": "string"}}
        )
        await self.collection.create_index([("status", 1), ("available_at", 1)])
        await self.collection.create_index([("document_id", 1), ("created_at", -1)])

    async def enqueue(
        self,
        kind: JobKind,
        document_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        # All operations on a document serialize through one active job.
        active_key = document_id
        job = {
            "job_id": uuid4().hex,
            "kind": kind,
            "document_id": document_id,
            "payload": payload or {},
            "status": "queued",
            "attempts": 0,
            "created_at": now,
            "available_at": now,
            "active_key": active_key,
            "history": [],
            "error": None,
        }
        try:
            await self.collection.insert_one(job)
        except DuplicateKeyError:
            existing = await self.collection.find_one({"active_key": active_key}, {"_id": 0})
            if existing:
                if existing["kind"] != kind or existing.get("payload", {}) != (payload or {}):
                    raise AppError(
                        "Another operation is active for this document.",
                        code="document_busy",
                        status_code=409,
                    ) from None
                return dict(existing)
            raise
        job.pop("_id", None)
        return job

    async def get(self, job_id: str) -> dict[str, Any] | None:
        return await self.collection.find_one({"job_id": job_id}, {"_id": 0})

    async def claim(self, owner: str) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        return await self.collection.find_one_and_update(
            {
                "$or": [
                    {"status": "queued", "available_at": {"$lte": now}},
                    {"status": "running", "lease_until": {"$lt": now}},
                ]
            },
            {
                "$set": {
                    "status": "running",
                    "owner": owner,
                    "started_at": now,
                    "lease_until": now + timedelta(seconds=get_settings().job_lease_seconds),
                },
                "$inc": {"attempts": 1},
                "$push": {"history": {"event": "claimed", "at": now, "owner": owner}},
            },
            sort=[("available_at", 1)],
            return_document=ReturnDocument.AFTER,
        )

    async def heartbeat(self, job_id: str, owner: str) -> bool:
        result = await self.collection.update_one(
            {"job_id": job_id, "owner": owner, "status": "running"},
            {
                "$set": {
                    "lease_until": datetime.now(UTC)
                    + timedelta(seconds=get_settings().job_lease_seconds)
                }
            },
        )
        return result.matched_count == 1

    async def finish(
        self,
        job: dict[str, Any],
        owner: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        retryable: bool = False,
    ) -> None:
        now = datetime.now(UTC)
        retry = (
            error is not None and retryable and job["attempts"] < get_settings().job_max_attempts
        )
        status = "queued" if retry else "failed" if error else "completed"
        update: dict[str, Any] = {
            "$set": {
                "status": status,
                "error": error,
                "result": result,
                "completed_at": None if retry else now,
                "available_at": now + timedelta(seconds=min(300, 2 ** job["attempts"])),
            },
            "$unset": {"owner": "", "lease_until": ""},
            "$push": {"history": {"event": status, "at": now, "error": error}},
        }
        if not retry:
            update["$unset"]["active_key"] = ""
        await self.collection.update_one(
            {"job_id": job["job_id"], "owner": owner, "status": "running"},
            update,
        )

    async def retry(self, job_id: str) -> dict[str, Any] | None:
        job = await self.get(job_id)
        if job is None:
            return None
        if job["status"] != "failed":
            raise ValueError("Only failed jobs can be retried.")
        # Preserve identity and completed extraction IDs on an explicit retry.
        try:
            updated = await self.collection.find_one_and_update(
                {"job_id": job_id, "status": "failed"},
                {
                    "$set": {
                        "status": "queued",
                        "active_key": job["document_id"],
                        "available_at": datetime.now(UTC),
                        "attempts": 0,
                        "error": None,
                    },
                    "$push": {"history": {"event": "manual_retry", "at": datetime.now(UTC)}},
                },
                return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError as exc:
            raise ValueError("A job is already active for this document.") from exc
        if updated is None:
            raise ValueError("Job was already retried.")
        updated.pop("_id", None)
        return dict(updated)
