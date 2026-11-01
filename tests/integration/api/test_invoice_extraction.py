from datetime import UTC, datetime
from unittest.mock import AsyncMock

import app.features.invoices.router as invoice_router
import pytest
from app.core.config import Settings
from app.core.database import get_database
from app.features.document_extraction.schemas import (
    DocumentResponse,
    DocumentStatus,
    DocumentType,
)
from app.features.invoices.router import get_invoice_repository
from app.features.invoices.schemas import (
    InvoiceAIResult,
    InvoiceData,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient

DOCUMENT_ID = "a" * 32


class FakeCursor:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self.results = results

    def limit(self, _: int) -> "FakeCursor":
        return self

    async def to_list(self, length: int) -> list[dict[str, object]]:
        return self.results[:length]


class FakeCollection:
    def __init__(
        self,
        *,
        document: DocumentResponse | None,
        source_results: list[dict[str, object]],
    ) -> None:
        self.document = document
        self.source_results = source_results

    async def find_one(
        self,
        query: dict[str, object],
        projection: dict[str, int] | None = None,
        sort: list[tuple[str, int]] | None = None,
    ) -> dict[str, object] | None:
        if "document_id" in query and self.document is not None:
            if query["document_id"] == self.document.document_id:
                return self.document.model_dump(mode="json")

        return None

    def find(
        self,
        query: dict[str, object],
        projection: dict[str, int],
    ) -> FakeCursor:
        results = [
            item
            for item in self.source_results
            if item.get("document_id") == query.get("document_id")
        ]

        unit_filter = query.get("unit_id")
        if isinstance(unit_filter, dict):
            requested = unit_filter.get("$in", [])
            results = [item for item in results if item.get("unit_id") in requested]

        return FakeCursor(results)


class FakeDatabase:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collection


class FakeInvoiceRepository:
    def __init__(self) -> None:
        self.save_extraction = AsyncMock()
        self.get_extraction = AsyncMock(return_value=None)
        self.get_latest_extraction = AsyncMock(return_value=None)
        self.find_duplicate_invoice = AsyncMock(return_value=None)


class FakeResponse:
    status = "completed"
    id = "response-1"
    output: list[object] = []

    def __init__(self, parsed: InvoiceAIResult) -> None:
        self.output_parsed = parsed


class FakeResponses:
    def __init__(self, parsed: InvoiceAIResult) -> None:
        self.parse = AsyncMock(return_value=FakeResponse(parsed))


class FakeAsyncOpenAI:
    def __init__(self, parsed: InvoiceAIResult, **_: object) -> None:
        self.responses = FakeResponses(parsed)
        self.options: dict[str, object] = {}

    async def __aenter__(self) -> "FakeAsyncOpenAI":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def with_options(self, **options: object) -> "FakeAsyncOpenAI":
        self.options = options
        return self


def _document(status: DocumentStatus = DocumentStatus.COMPLETED) -> DocumentResponse:
    return DocumentResponse(
        document_id=DOCUMENT_ID,
        filename="invoice.pdf",
        file_type=DocumentType.PDF,
        file_size=100,
        status=status,
        created_at=datetime.now(UTC),
    )


def _database(document: DocumentResponse | None) -> FakeDatabase:
    collection = FakeCollection(
        document=document,
        source_results=[
            {
                "document_id": DOCUMENT_ID,
                "unit_id": "page:1",
                "kind": "page",
                "status": "completed",
                "page_number": 1,
                "native_text": "Not an invoice document",
                "ocr_text": "",
                "text": "Not an invoice document",
                "extraction_method": "pymupdf",
            }
        ],
    )
    return FakeDatabase(collection)


def _configure_dependencies(
    app: FastAPI,
    database: FakeDatabase,
    repository: FakeInvoiceRepository,
) -> None:
    app.dependency_overrides[get_database] = lambda: database
    app.dependency_overrides[get_invoice_repository] = lambda: repository


def _clear_dependencies(app: FastAPI) -> None:
    app.dependency_overrides.pop(get_database, None)
    app.dependency_overrides.pop(get_invoice_repository, None)


@pytest.mark.integration
def test_invoice_extraction_runs_through_full_api_flow(
    app: FastAPI,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _database(_document())
    repository = FakeInvoiceRepository()
    fake_openai = FakeAsyncOpenAI(
        InvoiceAIResult(
            is_invoice=False,
            invoice=InvoiceData(),
            evidence=[],
        )
    )
    _configure_dependencies(app, database, repository)
    monkeypatch.setattr(
        invoice_router,
        "get_settings",
        lambda: Settings(
            openai_api_key="test-key",
            openai_model="gpt-test",
            environment="test",
            allowed_hosts=["*"],
        ),
    )
    monkeypatch.setattr(
        invoice_router,
        "AsyncOpenAI",
        lambda **kwargs: fake_openai,
    )
    enqueue = AsyncMock(return_value={"job_id": "b" * 32, "status": "queued"})
    monkeypatch.setattr("app.features.jobs.JobQueue.enqueue", enqueue)

    try:
        response = client.post(
            f"/api/v1/invoices/documents/{DOCUMENT_ID}/extract",
            json={"unit_ids": ["page:1"]},
        )
    finally:
        _clear_dependencies(app)

    assert response.status_code == 202
    payload = response.json()
    assert len(payload["job_id"]) == 32
    assert payload["status"] == "queued"
    enqueue.assert_awaited_once()
    fake_openai.responses.parse.assert_not_awaited()


@pytest.mark.integration
def test_invoice_extraction_returns_service_unavailable_without_api_key(
    app: FastAPI,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _database(_document())
    repository = FakeInvoiceRepository()
    _configure_dependencies(app, database, repository)
    monkeypatch.setattr(
        invoice_router,
        "get_settings",
        lambda: Settings(
            openai_api_key=None,
            environment="test",
            allowed_hosts=["*"],
        ),
    )

    try:
        response = client.post(
            f"/api/v1/invoices/documents/{DOCUMENT_ID}/extract",
            json={},
        )
    finally:
        _clear_dependencies(app)

    assert response.status_code == 503
    assert response.json()["error"]["message"] == ("OpenAI integration is not configured.")
    repository.save_extraction.assert_not_awaited()


@pytest.mark.integration
def test_invoice_extraction_rejects_document_not_ready(
    app: FastAPI,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _database(_document(DocumentStatus.PROCESSING))
    repository = FakeInvoiceRepository()
    _configure_dependencies(app, database, repository)
    monkeypatch.setattr(
        invoice_router,
        "get_settings",
        lambda: Settings(
            openai_api_key="test-key",
            environment="test",
            allowed_hosts=["*"],
        ),
    )

    try:
        response = client.post(
            f"/api/v1/invoices/documents/{DOCUMENT_ID}/extract",
            json={},
        )
    finally:
        _clear_dependencies(app)

    assert response.status_code == 409
    assert response.json()["error"]["message"] == (
        "Document processing must finish before invoice extraction."
    )


@pytest.mark.integration
def test_invoice_extraction_rejects_invalid_document_id(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/invoices/documents/not-a-document/extract",
        json={},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
