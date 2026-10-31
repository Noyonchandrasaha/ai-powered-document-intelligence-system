import asyncio
import io
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.features.document_extraction.schemas import (
    DocumentResponse,
    DocumentStatus,
    DocumentType,
)
from app.features.document_extraction.service import DocumentService
from fastapi import UploadFile


def create_upload_file(filename: str = "report.pdf") -> UploadFile:
    return UploadFile(file=io.BytesIO(b"file content"), filename=filename)


def create_document(document_id: str = "document-01") -> DocumentResponse:
    return DocumentResponse(
        document_id=document_id,
        filename="report.pdf",
        file_type=DocumentType.PDF,
        file_size=12,
        status=DocumentStatus.PENDING,
        created_at=datetime.now(UTC),
    )


def create_service() -> tuple[DocumentService, MagicMock]:
    repository = MagicMock()
    return DocumentService(repository), repository


@pytest.mark.unit
def test_upload_document_saves_validates_and_persists_file():
    service, repository = create_service()
    repository.create_document = AsyncMock()
    file_path = Path("uploads/document-01/source.pdf")

    with (
        patch("app.features.document_extraction.service.uuid4") as uuid,
        patch("app.features.document_extraction.service.validate_extension") as extension,
        patch("app.features.document_extraction.service.save_file", new_callable=AsyncMock) as save,
        patch("app.features.document_extraction.service.validate_document") as validate,
    ):
        uuid.return_value.hex = "document-01"
        extension.return_value = DocumentType.PDF
        save.return_value = file_path, 12
        validate.return_value = DocumentType.PDF, 12

        result = asyncio.run(service.upload_document(create_upload_file()))

    assert result.document_id == "document-01"
    assert result.file_type is DocumentType.PDF
    assert result.status is DocumentStatus.PENDING
    save.assert_awaited_once()
    validate.assert_called_once_with(file_path, "report.pdf")
    repository.create_document.assert_awaited_once_with(
        document=result,
        storage_path=str(file_path),
    )


@pytest.mark.unit
def test_upload_document_cleans_file_when_validation_fails():
    service, repository = create_service()
    repository.create_document = AsyncMock()
    file_path = Path("uploads/document-01/source.pdf")

    with (
        patch("app.features.document_extraction.service.uuid4") as uuid,
        patch("app.features.document_extraction.service.validate_extension") as extension,
        patch("app.features.document_extraction.service.save_file", new_callable=AsyncMock) as save,
        patch(
            "app.features.document_extraction.service.validate_document",
            side_effect=ValueError("invalid document"),
        ),
        patch("app.features.document_extraction.service.delete_file") as delete,
    ):
        uuid.return_value.hex = "document-01"
        extension.return_value = DocumentType.PDF
        save.return_value = file_path, 12

        with pytest.raises(ValueError, match="invalid document"):
            asyncio.run(service.upload_document(create_upload_file()))

    delete.assert_called_once_with("document-01")
    repository.create_document.assert_not_awaited()


@pytest.mark.unit
def test_upload_documents_rejects_empty_batch():
    service, _ = create_service()

    with pytest.raises(ValueError, match="At least one file"):
        asyncio.run(service.upload_documents([]))


@pytest.mark.unit
def test_upload_documents_rejects_batch_over_limit():
    service, _ = create_service()
    files = [create_upload_file() for _ in range(21)]

    with pytest.raises(ValueError, match="Maximum 20 files"):
        asyncio.run(service.upload_documents(files))


@pytest.mark.unit
def test_upload_documents_returns_batch_response():
    service, _ = create_service()
    first_document = create_document("document-01")
    second_document = create_document("document-02")
    service.upload_document = AsyncMock(side_effect=[first_document, second_document])

    result = asyncio.run(
        service.upload_documents([create_upload_file(), create_upload_file()])
    )

    assert result.total_files == 2
    assert result.documents == [first_document, second_document]
    assert result.batch_id
    assert service.upload_document.await_count == 2


@pytest.mark.unit
def test_upload_documents_rolls_back_previous_uploads_on_failure():
    service, _ = create_service()
    first_document = create_document("document-01")
    service.upload_document = AsyncMock(
        side_effect=[first_document, RuntimeError("upload failed")]
    )
    service.delete_document = AsyncMock(return_value=True)

    with pytest.raises(RuntimeError, match="upload failed"):
        asyncio.run(
            service.upload_documents([create_upload_file(), create_upload_file()])
        )

    service.delete_document.assert_awaited_once_with("document-01")


@pytest.mark.unit
def test_get_document_delegates_to_repository():
    service, repository = create_service()
    document = create_document()
    repository.get_document = AsyncMock(return_value=document)

    result = asyncio.run(service.get_document("document-01"))

    assert result == document
    repository.get_document.assert_awaited_once_with("document-01")


@pytest.mark.unit
def test_delete_document_returns_false_when_document_is_missing():
    service, repository = create_service()
    repository.get_document = AsyncMock(return_value=None)
    repository.delete_document = AsyncMock()

    assert asyncio.run(service.delete_document("missing")) is False
    repository.delete_document.assert_not_awaited()


@pytest.mark.unit
def test_delete_document_removes_metadata_and_file():
    service, repository = create_service()
    repository.get_document = AsyncMock(return_value=create_document())
    repository.delete_document = AsyncMock(return_value=True)

    with patch("app.features.document_extraction.service.delete_file") as delete:
        result = asyncio.run(service.delete_document("document-01"))

    assert result is True
    repository.delete_document.assert_awaited_once_with("document-01")
    delete.assert_called_once_with("document-01")
