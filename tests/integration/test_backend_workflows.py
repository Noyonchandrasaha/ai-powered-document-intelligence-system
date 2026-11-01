import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pymupdf
import pytest
from app.core.config import Settings
from app.core.database import get_database
from app.features.document_extraction.processing_service import DocumentProcessingService
from app.features.document_extraction.repository import DocumentRepository, ProcessingRepository
from app.features.document_extraction.schemas import DocumentResponse, DocumentStatus, DocumentType
from app.features.invoices.repository import InvoiceRepository
from app.features.invoices.schemas import (
    ExtractionStatus,
    InvoiceData,
    InvoiceExtractionResponse,
    InvoiceLineItem,
)
from app.features.invoices.segmentation import InvoiceGrouper, group_invoice_units
from app.features.jobs import JobQueue
from app.features.reconciliation import compare_invoices, reconcile
from app.main import create_app
from fastapi.testclient import TestClient
from mongomock_motor import AsyncMongoMockClient


def database():
    return AsyncMongoMockClient(tz_aware=True).test_backend


def invoice(number="INV-1", total="110.00"):
    return InvoiceData(
        invoice_number=number,
        invoice_date="2026-10-01",
        supplier_name="Supplier",
        customer_name="Customer",
        subtotal="100",
        tax="10",
        total=total,
        currency="USD",
        line_items=[
            InvoiceLineItem(description="Product", quantity="2", unit_price="50", amount="100")
        ],
    )


def result(document_id="a" * 32, **kwargs):
    return InvoiceExtractionResponse(
        document_id=document_id,
        invoice=invoice(**kwargs),
        status="completed",
        evidence=[],
        extracted_at=datetime.now(UTC),
    )


def page(number, text):
    return {
        "document_id": "a" * 32,
        "unit_id": f"page:{number}",
        "kind": "page",
        "page_number": number,
        "text": text,
        "status": "completed",
        "metadata": {},
    }


@pytest.mark.integration
def test_queue_atomic_claim_expired_lease_fencing_and_retries(monkeypatch):
    async def run():
        db = database()
        queue = JobQueue(db)
        await queue.ensure_indexes()
        job = await queue.enqueue("process", "a" * 32)
        duplicate = await queue.enqueue("process", "a" * 32)
        assert duplicate["job_id"] == job["job_id"]
        claimed = await queue.claim("worker-1")
        assert claimed["attempts"] == 1
        assert await queue.claim("worker-2") is None
        await db.jobs.update_one(
            {"job_id": job["job_id"]},
            {"$set": {"lease_until": datetime.now(UTC) - timedelta(seconds=1)}},
        )
        recovered = await queue.claim("worker-2")
        assert recovered["attempts"] == 2
        assert not await queue.heartbeat(job["job_id"], "worker-1")
        assert await queue.heartbeat(job["job_id"], "worker-2")
        await queue.finish(claimed, "worker-1", result={"wrong": True})
        assert (await queue.get(job["job_id"]))["status"] == "running"
        await queue.finish(recovered, "worker-2", error="timeout", retryable=True)
        assert (await queue.get(job["job_id"]))["status"] == "queued"
        await db.jobs.update_one(
            {"job_id": job["job_id"]},
            {"$set": {"available_at": datetime.now(UTC) - timedelta(seconds=1)}},
        )
        last = await queue.claim("worker-3")
        await queue.finish(last, "worker-3", error="timeout", retryable=True)
        assert (await queue.get(job["job_id"]))["status"] == "failed"
        retried = await queue.retry(job["job_id"])
        assert retried["job_id"] == job["job_id"]
        assert retried["attempts"] == 0
        with pytest.raises(ValueError):
            await queue.retry(job["job_id"])
        assert await queue.retry("missing") is None

    asyncio.run(run())


@pytest.mark.integration
def test_correction_revalidates_audits_and_invalidates_embedding():
    async def run():
        db = database()
        repo = InvoiceRepository(db)
        await repo.ensure_indexes()
        await repo.save_extraction(
            extraction_id="b" * 32,
            result=result(),
            model_name="test",
            prompt_version="test",
            source_units=("page:1",),
            embedding=[1, 0],
        )
        corrected = await repo.correct_extraction(
            "b" * 32, invoice(total="999"), corrected_by="reviewer", reason="test correction"
        )
        assert corrected.status.value == "needs_review"
        assert corrected.validation_errors
        stored = await db.invoice_extractions.find_one({"extraction_id": "b" * 32})
        assert "embedding" not in stored
        assert stored["correction_history"][0]["invoice_before"]["total"] == "110.00"
        empty = await repo.correct_extraction(
            "b" * 32, InvoiceData(), corrected_by="reviewer", reason="missing"
        )
        assert "total" in empty.missing_fields
        fixed = await repo.correct_extraction(
            "b" * 32, invoice(), corrected_by="reviewer", reason="fix"
        )
        assert fixed.status.value == "completed"
        assert (
            await repo.correct_extraction("missing", invoice(), corrected_by="x", reason="x")
            is None
        )

    asyncio.run(run())


@pytest.mark.integration
def test_supplier_scoped_duplicate_identity_is_atomic():
    async def run():
        repo = InvoiceRepository(database())
        await repo.ensure_indexes()
        a, b = await asyncio.gather(
            repo.find_duplicate_invoice("doc-a", " INV-1 ", "Supplier"),
            repo.find_duplicate_invoice("doc-b", "inv-1", " supplier "),
        )
        assert sum(item is not None for item in (a, b)) == 1
        assert await repo.find_duplicate_invoice("doc-c", "inv-1", "Other Supplier") is None

    asyncio.run(run())


@pytest.mark.integration
def test_pipeline_persists_progress_and_handles_real_thousand_page_pdf(
    workspace_tmp_path, monkeypatch
):
    async def run():
        path = workspace_tmp_path / "large.pdf"
        with pymupdf.open() as pdf:
            for number in range(1000):
                pdf.new_page().insert_text((72, 72), f"Invoice No: INV-{number}\nTotal USD 110")
            pdf.save(path)
        db = database()
        doc = DocumentResponse(
            document_id="a" * 32,
            filename="large.pdf",
            file_type=DocumentType.PDF,
            file_size=path.stat().st_size,
            status=DocumentStatus.PENDING,
            created_at=datetime.now(UTC),
        )
        await DocumentRepository(db).create_document(doc, str(path))
        service = DocumentProcessingService(db)
        assert await service.process_document(doc.document_id) == DocumentStatus.COMPLETED
        status = await DocumentRepository(db).get_document_status(doc.document_id)
        assert status.processed_units == 1000
        assert status.total_pages == 1000
        units = await ProcessingRepository(db).get_results(doc.document_id)
        assert len(list(group_invoice_units(sorted(units, key=lambda u: u["page_number"])))) == 1000
        assert await db.extraction_results.count_documents({}) == 1000
        assert await DocumentRepository(db).delete_document(doc.document_id)
        assert await db.extraction_results.count_documents({}) == 0

    asyncio.run(run())


@pytest.mark.integration
def test_all_failed_units_are_failed_and_progress_survives(monkeypatch):
    async def failed(*args, **kwargs):
        yield {**page(1, ""), "status": "failed", "error": "OCR unavailable"}

    monkeypatch.setattr(
        "app.features.document_extraction.processing_service.process_document", failed
    )

    async def run():
        db = database()
        doc = DocumentResponse(
            document_id="a" * 32,
            filename="x.pdf",
            file_type="pdf",
            file_size=1,
            status="pending",
            created_at=datetime.now(UTC),
        )
        await DocumentRepository(db).create_document(doc, "missing.pdf")
        assert (
            await DocumentProcessingService(db).process_document(doc.document_id)
            == DocumentStatus.FAILED
        )
        assert (
            await DocumentRepository(db).get_document_status(doc.document_id)
        ).processed_units == 1

    asyncio.run(run())


@pytest.mark.integration
def test_search_static_route_units_and_job_api(app, client):
    db = database()
    app.dependency_overrides[get_database] = lambda: db
    response = client.get("/api/v1/documents/search")
    assert response.status_code == 200
    assert response.json() == []
    assert client.get(f"/api/v1/documents/{'a' * 32}/units").status_code == 404
    assert client.get(f"/api/v1/jobs/{'a' * 32}").status_code == 404
    assert client.post(f"/api/v1/jobs/{'a' * 32}/retry").status_code == 404
    app.dependency_overrides.clear()


@pytest.mark.integration
def test_api_authentication_rate_limit_and_cors():
    app = create_app(
        Settings(
            environment="test",
            api_key="test-secret",
            rate_limit_per_minute=2,
            allowed_origins=["http://localhost:3000"],
            allowed_hosts=["testserver"],
        )
    )
    with TestClient(app) as client:
        assert client.get("/api/v1/jobs/" + "a" * 32).status_code == 401
        # Invalid paths still exercise authenticated requests without needing Mongo.
        headers = {"X-API-Key": "test-secret"}
        assert client.get("/api/v1/unknown", headers=headers).status_code == 404
        assert client.get("/api/v1/unknown", headers=headers).status_code == 404
        assert client.get("/api/v1/unknown", headers=headers).status_code == 429
        response = client.options(
            "/api/v1/documents/search",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-API-Key",
            },
        )
        assert response.status_code == 200


@pytest.mark.unit
def test_grouping_continuations_ambiguous_pages_and_register_chunks():
    groups = list(
        group_invoice_units(
            [page(1, "Invoice No: INV-1"), page(2, "continuation"), page(3, "Invoice No: INV-2")]
        )
    )
    assert [g["unit_ids"] for g in groups] == [["page:1", "page:2"], ["page:3"]]
    assert not groups[0]["requires_review"]
    assert list(group_invoice_units([page(1, "Invoice No: A\nInvoice No: B")]))[0][
        "requires_review"
    ]
    grouper = InvoiceGrouper()
    header = {
        "row_number": 1,
        "cells": [
            {"coordinate": "A1", "value": "Invoice Number"},
            {"coordinate": "B1", "value": "total"},
        ],
    }
    row = {"row_number": 2, "cells": [{"coordinate": "A2", "value": "INV-1"}]}
    unit = {
        "kind": "sheet_chunk",
        "sheet_index": 1,
        "unit_id": "sheet:1:row:1",
        "rows": [header, row],
    }
    assert len(list(grouper.feed(unit))) == 1
    second = list(
        grouper.feed({**unit, "unit_id": "sheet:1:row:3", "rows": [{**row, "row_number": 3}]})
    )
    assert second[0]["units"][0]["rows"][0] == header


@pytest.mark.unit
def test_grouping_multiline_invoice_labels_and_purchase_order_boundary():
    def labeled_page(number: int, invoice_number: str) -> dict[str, object]:
        return page(
            number,
            "INVOICE\nINVOICE NUMBER\nINVOICE DATE\n2026-10-20\n"
            f"{invoice_number}\nSupplier Ltd.\nTotal 100.00",
        )

    groups = list(
        group_invoice_units(
            [
                labeled_page(1, "EDGE-2026-001"),
                labeled_page(2, "EDGE-2026-002"),
                labeled_page(3, "EDGE-2026-001"),
                page(
                    4,
                    "INVOICE\nINVOICE NUMBER\nINVOICE DATE\n05/06/2026\n"
                    "EDGE-2026-004\nSupplier Ltd.",
                ),
                page(5, "PURCHASE ORDER\nPO-2026-009\nNot an invoice."),
            ]
        )
    )

    assert [group["invoice_hint"] for group in groups] == [
        "edge-2026-001",
        "edge-2026-002",
        "edge-2026-001",
        "edge-2026-004",
        None,
    ]
    assert [group["unit_ids"] for group in groups] == [
        ["page:1"],
        ["page:2"],
        ["page:3"],
        ["page:4"],
        ["page:5"],
    ]
    assert groups[-1]["requires_review"]


@pytest.mark.unit
def test_reconciliation_does_not_pass_missing_ambiguous_or_different_values():
    left, right = result(), result("b" * 32)
    assert compare_invoices(left, right)["status"] == "matched"
    right.invoice.total = "111"
    assert compare_invoices(left, right)["status"] == "mismatch"
    right.invoice = InvoiceData()
    assert compare_invoices(left, right)["status"] == "mismatch"
    assert reconcile([left], [], Decimal("0.01"))[0]["status"] == "unmatched"
    assert reconcile([left, left], [left], Decimal("0.01"))[0]["status"] == "needs_review"
    right = result("b" * 32)
    right.status = ExtractionStatus.NEEDS_REVIEW
    assert compare_invoices(left, right)["status"] == "needs_review"
    right.invoice.line_items *= 2
    assert compare_invoices(left, right)["status"] == "mismatch"


@pytest.mark.integration
def test_semantic_search_checks_all_candidates_without_fixed_cap():
    async def run():
        db = database()
        repo = InvoiceRepository(db)
        for index, vector in enumerate(([0, 1], [1, 0], [-1, 0])):
            await repo.save_extraction(
                extraction_id=str(index),
                result=result(),
                model_name="test",
                prompt_version="test",
                source_units=(),
                embedding=vector,
            )
        matches = await repo.semantic_search([1, 0], limit=1)
        assert matches[0].extraction_id == "1"

    asyncio.run(run())
