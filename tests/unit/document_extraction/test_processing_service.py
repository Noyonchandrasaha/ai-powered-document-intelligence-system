import asyncio
from datetime import UTC, datetime
from functools import wraps
from unittest.mock import AsyncMock, MagicMock

import pytest
from app.features.document_extraction import processing_service as module
from app.features.document_extraction.processing_service import (
    DocumentProcessingService,
)
from app.features.document_extraction.schemas import (
    DocumentResponse,
    DocumentStatus,
    DocumentType,
)


def _run_async(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


def _document() -> DocumentResponse:
    return DocumentResponse(
        document_id="doc-1",
        filename="invoice.pdf",
        file_type=DocumentType.PDF,
        file_size=100,
        status=DocumentStatus.PENDING,
        created_at=datetime.now(UTC),
    )


def _service() -> tuple[DocumentProcessingService, MagicMock, MagicMock]:
    documents = MagicMock()
    documents.get_document = AsyncMock(return_value=_document())
    documents.get_storage_path = AsyncMock(return_value="source.pdf")
    documents.update_document_status = AsyncMock()
    documents.update_progress = AsyncMock()

    processing = MagicMock()
    processing.replace_results = AsyncMock()
    processing.clear_results = AsyncMock()
    processing.save_result = AsyncMock()
    processing.add_history = AsyncMock()

    service = DocumentProcessingService.__new__(DocumentProcessingService)
    service.documents = documents
    service.processing = processing
    return service, documents, processing


@pytest.mark.unit
@_run_async
async def test_processing_service_persists_results_and_completed_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, documents, processing = _service()

    async def fake_process(*args, **kwargs):
        yield {
            "document_id": "doc-1",
            "unit_id": "page:1",
            "status": "completed",
            "metadata": {"total_pages": 1},
            "error": None,
        }

    monkeypatch.setattr(module, "process_document", fake_process)
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: MagicMock(google_vision_credentials_file="missing.json"),
    )

    status = await service.process_document("doc-1")

    assert status is DocumentStatus.COMPLETED
    documents.update_document_status.assert_any_await(
        "doc-1",
        DocumentStatus.PROCESSING,
    )
    documents.update_document_status.assert_any_await(
        "doc-1",
        DocumentStatus.COMPLETED,
        total_pages=1,
        error_message=None,
    )
    processing.save_result.assert_awaited_once()
    assert processing.add_history.await_count == 2


@pytest.mark.unit
@_run_async
async def test_processing_service_marks_failed_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, documents, processing = _service()

    async def fake_process(*args, **kwargs):
        raise RuntimeError("parser failed")
        yield  # pragma: no cover

    monkeypatch.setattr(module, "process_document", fake_process)
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: MagicMock(google_vision_credentials_file="missing.json"),
    )

    status = await service.process_document("doc-1")

    assert status is DocumentStatus.FAILED
    documents.update_document_status.assert_any_await(
        "doc-1",
        DocumentStatus.FAILED,
        error_message="RuntimeError: parser failed",
    )
    assert processing.add_history.await_count == 2
