import io
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pymupdf
import pytest
from app.features.document_extraction.router import get_document_service
from app.features.document_extraction.schemas import (
    DocumentResponse,
    DocumentStatus,
    DocumentType,
    UploadResponse,
)
from app.features.document_extraction.service import DocumentService
from app.storage import local
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import Workbook
from PIL import Image


def create_document() -> DocumentResponse:
    return DocumentResponse(
        document_id="a" * 32,
        filename="report.pdf",
        file_type=DocumentType.PDF,
        file_size=12,
        status=DocumentStatus.PENDING,
        created_at=datetime.now(UTC),
    )


def create_pdf_bytes() -> bytes:
    document = pymupdf.open()
    document.new_page()
    content = document.tobytes()
    document.close()
    return content


def create_jpg_bytes() -> bytes:
    content = io.BytesIO()
    Image.new("RGB", (10, 10), color="white").save(content, format="JPEG")
    return content.getvalue()


def create_xlsx_bytes() -> bytes:
    content = io.BytesIO()
    workbook = Workbook()
    workbook.active["A1"] = "amount"
    workbook.save(content)
    return content.getvalue()


class InMemoryDocumentRepository:
    def __init__(self) -> None:
        self.documents: dict[str, DocumentResponse] = {}

    async def create_document(
        self,
        document: DocumentResponse,
        storage_path: str,
    ) -> DocumentResponse:
        self.documents[document.document_id] = document
        return document

    async def get_document(self, document_id: str) -> DocumentResponse | None:
        return self.documents.get(document_id)

    async def delete_document(self, document_id: str) -> bool:
        return self.documents.pop(document_id, None) is not None


@pytest.fixture
def real_document_service(app: FastAPI, workspace_tmp_path, monkeypatch: pytest.MonkeyPatch):
    upload_dir = workspace_tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(local, "UPLOAD_DIR", upload_dir)

    service = DocumentService(InMemoryDocumentRepository())
    app.dependency_overrides[get_document_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_document_service, None)


@pytest.fixture
def document_service(app: FastAPI):
    service = MagicMock()
    app.dependency_overrides[get_document_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_document_service, None)


@pytest.mark.integration
def test_upload_documents_returns_batch_response(
    client: TestClient,
    document_service: MagicMock,
):
    document = create_document()
    document_service.upload_documents = AsyncMock(
        return_value=UploadResponse(
            batch_id="b" * 32,
            total_files=1,
            documents=[document],
        )
    )

    response = client.post(
        "/api/v1/documents/upload",
        files=[("files", ("report.pdf", b"file content", "application/pdf"))],
    )

    assert response.status_code == 202
    assert response.json()["total_files"] == 1
    document_service.upload_documents.assert_awaited_once()


@pytest.mark.integration
def test_upload_documents_processes_multiple_supported_files(
    client: TestClient,
    real_document_service: DocumentService,
):
    response = client.post(
        "/api/v1/documents/upload",
        files=[
            ("files", ("invoice.pdf", create_pdf_bytes(), "application/pdf")),
            ("files", ("receipt.jpg", create_jpg_bytes(), "image/jpeg")),
            (
                "files",
                (
                    "financial_report.xlsx",
                    create_xlsx_bytes(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ),
            ),
        ],
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["batch_id"]
    assert payload["total_files"] == 3
    assert {document["filename"] for document in payload["documents"]} == {
        "invoice.pdf",
        "receipt.jpg",
        "financial_report.xlsx",
    }
    assert {document["file_type"] for document in payload["documents"]} == {
        "pdf",
        "jpg",
        "xlsx",
    }
    assert all(document["status"] == "pending" for document in payload["documents"])
    assert all(document["file_size"] > 0 for document in payload["documents"])
    assert all(document["total_pages"] is None for document in payload["documents"])
    assert len(real_document_service.repository.documents) == 3


@pytest.mark.integration
def test_upload_documents_returns_bad_request_for_invalid_file(
    client: TestClient,
    document_service: MagicMock,
):
    document_service.upload_documents = AsyncMock(
        side_effect=ValueError("Invalid document")
    )

    response = client.post(
        "/api/v1/documents/upload",
        files=[("files", ("report.pdf", b"file content", "application/pdf"))],
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Invalid document"


@pytest.mark.integration
def test_get_document_returns_document(
    client: TestClient,
    document_service: MagicMock,
):
    document = create_document()
    document_service.get_document = AsyncMock(return_value=document)

    response = client.get(f"/api/v1/documents/{document.document_id}")

    assert response.status_code == 200
    assert response.json()["document_id"] == document.document_id
    document_service.get_document.assert_awaited_once_with(document.document_id)


@pytest.mark.integration
def test_get_document_returns_not_found(client: TestClient, document_service: MagicMock):
    document_service.get_document = AsyncMock(return_value=None)

    response = client.get(f"/api/v1/documents/{'b' * 32}")

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Document not found."


@pytest.mark.integration
def test_get_document_rejects_invalid_id(client: TestClient, document_service: MagicMock):
    response = client.get("/api/v1/documents/not-a-document-id")

    assert response.status_code == 422
    document_service.get_document.assert_not_called()


@pytest.mark.integration
def test_delete_document_returns_no_content(
    client: TestClient,
    document_service: MagicMock,
):
    document_service.delete_document = AsyncMock(return_value=True)

    response = client.delete(f"/api/v1/documents/{'c' * 32}")

    assert response.status_code == 204
    assert response.content == b""


@pytest.mark.integration
def test_delete_document_returns_not_found(
    client: TestClient,
    document_service: MagicMock,
):
    document_service.delete_document = AsyncMock(return_value=False)

    response = client.delete(f"/api/v1/documents/{'d' * 32}")

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Document not found."
