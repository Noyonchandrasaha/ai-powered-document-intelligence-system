import asyncio
from datetime import UTC, datetime
from functools import wraps
from unittest.mock import AsyncMock, MagicMock

import pytest
from app.features.invoices.extractor import OpenAIInvoiceExtractor
from app.features.invoices.schemas import (
    ExtractionStatus,
    FieldEvidence,
    FieldStatus,
    InvoiceAIResult,
    InvoiceData,
    InvoiceExtractionResponse,
)
from app.features.invoices.service import (
    InvoiceService,
    InvoiceServiceError,
)


def _run_async(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


def _source_results() -> list[dict[str, object]]:
    return [
        {
            "document_id": "doc-1",
            "unit_id": "page:2",
            "kind": "page",
            "status": "completed",
            "page_number": 2,
            "native_text": "Invoice page two",
            "ocr_text": "",
            "text": "Invoice page two",
            "extraction_method": "pymupdf",
        },
        {
            "document_id": "doc-1",
            "unit_id": "page:1",
            "kind": "page",
            "status": "completed",
            "page_number": 1,
            "native_text": "Invoice page one",
            "ocr_text": "",
            "text": "Invoice page one",
            "extraction_method": "pymupdf",
        },
    ]


def _not_invoice_result() -> InvoiceAIResult:
    return InvoiceAIResult(
        is_invoice=False,
        invoice=InvoiceData(),
        evidence=[],
    )


def _stored_result(document_id: str = "doc-1") -> InvoiceExtractionResponse:
    return InvoiceExtractionResponse(
        document_id=document_id,
        status=ExtractionStatus.COMPLETED,
        invoice=InvoiceData(),
        evidence=[],
        extracted_at=datetime.now(UTC),
    )


def _service(
    *,
    ai_result: InvoiceAIResult | None = None,
    existing: InvoiceExtractionResponse | None = None,
) -> tuple[InvoiceService, MagicMock, MagicMock]:
    repository = MagicMock()
    repository.get_extraction = AsyncMock(return_value=existing)
    repository.get_latest_extraction = AsyncMock(return_value=None)
    repository.find_duplicate_invoice = AsyncMock(return_value=None)
    repository.save_extraction = AsyncMock()

    extractor = MagicMock(spec=OpenAIInvoiceExtractor)
    extractor.model = "gpt-test"
    extractor.extract = AsyncMock(return_value=ai_result or _not_invoice_result())

    return InvoiceService(repository, extractor), repository, extractor


@pytest.mark.unit
def test_prepare_context_sorts_units_and_preserves_incomplete_state() -> None:
    service, _, _ = _service()

    context = service._prepare_context(
        "doc-1",
        _source_results(),
        source_incomplete=True,
    )

    assert context.source_units == ("page:1", "page:2")
    assert context.incomplete is True
    assert "Original document processing was incomplete." in context.warnings


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source_results", "message"),
    [
        (
            [
                {
                    "document_id": "doc-1",
                    "unit_id": "page:1",
                    "kind": "page",
                    "status": "completed",
                    "page_number": 1,
                    "native_text": "text",
                },
                {
                    "document_id": "doc-1",
                    "unit_id": "page:1",
                    "kind": "page",
                    "status": "completed",
                    "page_number": 1,
                    "native_text": "text",
                },
            ],
            "Duplicate source unit",
        ),
        (
            [
                {
                    "document_id": "other",
                    "unit_id": "page:1",
                }
            ],
            "Source belongs to another document",
        ),
    ],
)
def test_prepare_context_rejects_invalid_sources(
    source_results: list[dict[str, object]],
    message: str,
) -> None:
    service, _, _ = _service()

    with pytest.raises(InvoiceServiceError, match=message):
        service._prepare_context("doc-1", source_results)


@pytest.mark.unit
@_run_async
async def test_extract_invoice_runs_validate_and_save_workflow() -> None:
    service, repository, extractor = _service()

    outcome = await service.extract_invoice(
        document_id="doc-1",
        source_results=_source_results(),
        extraction_id="ext-1",
    )

    assert outcome.extraction_id == "ext-1"
    assert outcome.result.status is ExtractionStatus.NOT_INVOICE
    extractor.extract.assert_awaited_once()
    repository.save_extraction.assert_awaited_once()
    assert repository.save_extraction.await_args.kwargs["model_name"] == ("gpt-test")


@pytest.mark.unit
@_run_async
async def test_extract_invoice_marks_incomplete_non_invoice_for_review() -> None:
    service, repository, _ = _service()

    outcome = await service.extract_invoice(
        document_id="doc-1",
        source_results=_source_results(),
        source_incomplete=True,
    )

    assert outcome.result.status is ExtractionStatus.NEEDS_REVIEW
    assert outcome.result.uncertain_fields == ["is_invoice"]
    repository.save_extraction.assert_awaited_once()


@pytest.mark.unit
@_run_async
async def test_extract_invoice_flags_duplicate_invoice_number() -> None:
    service, repository, _ = _service(
        ai_result=InvoiceAIResult(
            is_invoice=True,
            invoice=InvoiceData(invoice_number="INV-100"),
            evidence=[
                FieldEvidence(
                    field_path="invoice_number",
                    status=FieldStatus.EXTRACTED,
                    source_text="INV-100",
                    page_number=1,
                )
            ],
        )
    )
    repository.find_duplicate_invoice.return_value = {
        "document_id": "other-document",
        "extraction_id": "other-extraction",
    }

    source_results = _source_results()
    source_results[1]["native_text"] = "INV-100"
    source_results[1]["text"] = "INV-100"

    outcome = await service.extract_invoice(
        document_id="doc-1",
        source_results=source_results,
    )

    assert outcome.result.status is ExtractionStatus.NEEDS_REVIEW
    assert "invoice_number" in outcome.result.uncertain_fields
    assert "already exists" in outcome.result.validation_errors[0]


@pytest.mark.unit
@_run_async
async def test_extract_invoice_returns_existing_result_idempotently() -> None:
    existing = _stored_result()
    service, repository, extractor = _service(existing=existing)

    outcome = await service.extract_invoice(
        document_id="doc-1",
        source_results=[],
        extraction_id="ext-1",
    )

    assert outcome.result == existing
    extractor.extract.assert_not_awaited()
    repository.save_extraction.assert_not_awaited()


@pytest.mark.unit
@_run_async
async def test_extract_invoice_rejects_existing_result_for_other_document() -> None:
    service, _, _ = _service(existing=_stored_result("other"))

    with pytest.raises(
        InvoiceServiceError,
        match="Extraction ID belongs to another document",
    ):
        await service.extract_invoice(
            document_id="doc-1",
            source_results=[],
            extraction_id="ext-1",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "extraction_id",
    ["", " ", "x" * 129],
)
@_run_async
async def test_extract_invoice_rejects_invalid_extraction_id(
    extraction_id: str,
) -> None:
    service, repository, _ = _service()

    with pytest.raises(InvoiceServiceError, match="Invalid extraction ID"):
        await service.extract_invoice(
            document_id="doc-1",
            source_results=_source_results(),
            extraction_id=extraction_id,
        )

    repository.get_extraction.assert_not_awaited()


@pytest.mark.unit
@_run_async
async def test_get_methods_validate_and_delegate() -> None:
    service, repository, _ = _service()
    expected = _stored_result()
    repository.get_extraction.return_value = expected
    repository.get_latest_extraction.return_value = expected

    assert await service.get_extraction("ext-1") == expected
    assert await service.get_latest_extraction("doc-1") == expected
    repository.get_extraction.assert_awaited_once_with("ext-1")
    repository.get_latest_extraction.assert_awaited_once_with("doc-1")

    with pytest.raises(InvoiceServiceError, match="Extraction ID is required"):
        await service.get_extraction(" ")

    with pytest.raises(InvoiceServiceError, match="Document ID is required"):
        await service.get_latest_extraction(" ")
