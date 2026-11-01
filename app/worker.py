"""Run with python -m app.worker; scale by running additional worker processes."""

import asyncio
import contextlib
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from openai import AsyncOpenAI
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import get_settings
from app.core.database import close_mongo_client, get_database
from app.core.logging import configure_logging, get_logger
from app.features.document_extraction.processing_service import DocumentProcessingService
from app.features.document_extraction.repository import DocumentRepository, ProcessingRepository
from app.features.document_extraction.schemas import DocumentStatus
from app.features.invoices.context import InvoiceContextError
from app.features.invoices.extractor import (
    InvoiceExtractionError,
    InvoiceExtractionTemporaryError,
    OpenAIInvoiceExtractor,
)
from app.features.invoices.repository import InvoiceRepository
from app.features.invoices.segmentation import InvoiceGrouper
from app.features.invoices.service import InvoiceService
from app.features.jobs import JobQueue

logger = get_logger(__name__)


async def extract_job(database: AsyncDatabase[Any], job: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("APP_OPENAI_API_KEY is required for invoice extraction.")
    document_id = job["document_id"]
    document = await DocumentRepository(database).get_document(document_id)
    if document is None or document.status not in {
        DocumentStatus.COMPLETED,
        DocumentStatus.PARTIAL,
    }:
        raise ValueError("Document processing must finish before invoice extraction.")
    query: dict[str, Any] = {"document_id": document_id}
    selected = job["payload"].get("unit_ids")
    if selected:
        query["unit_id"] = {"$in": selected}
    cursor = (
        database["extraction_results"]
        .find(query, {"_id": 0})
        .sort([("page_number", 1), ("sheet_index", 1), ("start_row", 1)])
    )
    grouper = InvoiceGrouper()
    selected_units: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    counter = 0
    async with AsyncOpenAI(api_key=settings.openai_api_key) as client:
        service = InvoiceService(
            InvoiceRepository(database),
            OpenAIInvoiceExtractor(
                client, model=settings.openai_model, embedding_model=settings.openai_embedding_model
            ),
            semantic_search_enabled=settings.semantic_search_enabled,
        )

        async def extract_group(group: dict[str, Any]) -> None:
            nonlocal counter
            counter += 1
            extraction_id = (job["payload"].get("extraction_id") if selected else None) or uuid5(
                NAMESPACE_URL, f"{job['job_id']}:{counter}"
            ).hex
            candidate = {key: value for key, value in group.items() if key != "units"}
            candidate.update(
                document_id=document_id,
                job_id=job["job_id"],
                candidate_id=f"{job['job_id']}:{counter}",
                extraction_id=extraction_id,
            )
            await database["invoice_candidates"].replace_one(
                {"candidate_id": candidate["candidate_id"]},
                candidate,
                upsert=True,
            )
            try:
                outcome = await service.extract_invoice(
                    document_id=document_id,
                    source_results=group["units"],
                    source_incomplete=(
                        document.status == DocumentStatus.PARTIAL or group["requires_review"]
                    ),
                    extraction_id=extraction_id,
                )
                outcomes.append(
                    {"extraction_id": outcome.extraction_id, "status": outcome.result.status.value}
                )
            except InvoiceExtractionTemporaryError:
                raise
            except (InvoiceContextError, InvoiceExtractionError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                await database["invoice_candidates"].update_one(
                    {"candidate_id": candidate["candidate_id"]},
                    {"$set": {"status": "failed", "error": error}},
                )
                outcomes.append(
                    {"extraction_id": extraction_id, "status": "failed", "error": error}
                )
            await database["jobs"].update_one(
                {"job_id": job["job_id"]}, {"$set": {"processed_candidates": counter}}
            )

        async for unit in cursor:
            if selected:
                selected_units.append(unit)
            else:
                for group in grouper.feed(unit):
                    await extract_group(group)
        if selected:
            if {unit["unit_id"] for unit in selected_units} != set(selected):
                raise ValueError("Selected source units changed or disappeared.")
            await extract_group(
                {
                    "unit_ids": selected,
                    "units": selected_units,
                    "requires_review": False,
                    "invoice_hint": None,
                }
            )
        else:
            last = grouper.flush()
            if last:
                await extract_group(last)
    if not outcomes:
        raise ValueError("No invoice source units found.")
    return {
        "extractions": outcomes,
        "failed_candidates": sum(item["status"] == "failed" for item in outcomes),
    }


async def execute_job(database: AsyncDatabase[Any], job: dict[str, Any]) -> dict[str, Any]:
    if job["kind"] == "extract" or job.get("stage") == "extract":
        return await extract_job(database, job)
    status = await DocumentProcessingService(database).process_document(job["document_id"])
    if status == DocumentStatus.FAILED:
        raise RuntimeError("Document processing failed; inspect document history and source units.")
    if get_settings().openai_api_key:
        await database["jobs"].update_one(
            {"job_id": job["job_id"], "owner": job["owner"]},
            {"$set": {"stage": "extract", "document_status": status.value}},
        )
        return await extract_job(database, job)
    return {"document_status": status.value}


async def run_claimed(database: AsyncDatabase[Any], job: dict[str, Any], owner: str) -> None:
    queue = JobQueue(database)
    task = asyncio.create_task(execute_job(database, job))

    async def heartbeat() -> None:
        try:
            while True:
                await asyncio.sleep(get_settings().job_lease_seconds / 3)
                if not await queue.heartbeat(job["job_id"], owner):
                    task.cancel()
                    return
        except Exception:
            task.cancel()
            raise

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        if job["attempts"] > get_settings().job_max_attempts:
            task.cancel()
            raise ValueError("Maximum recovery attempts exceeded.")
        result = await task
        await queue.finish(
            job,
            owner,
            result=result,
            error="Some invoice groups failed; inspect candidates and retry."
            if result.get("failed_candidates")
            else None,
        )
    except asyncio.CancelledError:
        task.cancel()
        raise
    except Exception as exc:
        logger.exception("job_failed", job_id=job["job_id"])
        await queue.finish(
            job,
            owner,
            error=f"{type(exc).__name__}: {exc}",
            retryable=isinstance(exc, (InvoiceExtractionTemporaryError, OSError)),
        )
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


async def main() -> None:
    configure_logging(get_settings().log_level)
    database = get_database()
    queue = JobQueue(database)
    await queue.ensure_indexes()
    await DocumentRepository(database).ensure_indexes()
    await ProcessingRepository(database).ensure_indexes()
    await InvoiceRepository(database).ensure_indexes()
    owner = uuid4().hex
    try:
        while True:
            job = await queue.claim(owner)
            if job is None:
                # Upload metadata is the outbox: recover a crash between saving
                # metadata and inserting its queue record.
                async for document in database["documents"].find({"status": "pending"}):
                    await queue.enqueue("process", document["document_id"])
                await asyncio.sleep(get_settings().worker_poll_seconds)
                continue
            await run_claimed(database, job, owner)
    finally:
        await close_mongo_client()


if __name__ == "__main__":
    asyncio.run(main())
