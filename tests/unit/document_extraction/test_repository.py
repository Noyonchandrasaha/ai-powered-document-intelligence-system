import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from app.features.document_extraction.repository import DocumentRepository
from app.features.document_extraction.schemas import (
    DocumentResponse,
    DocumentStatus,
    DocumentType,
)


def create_document() -> DocumentResponse:
    return DocumentResponse(
        document_id="document-01",
        filename="report.pdf",
        file_type=DocumentType.PDF,
        file_size=1024,
        status=DocumentStatus.PENDING,
        created_at=datetime.now(UTC),
    )


def create_repository() -> tuple[DocumentRepository, MagicMock]:
    database = MagicMock()
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=None)
    collection.delete_many = AsyncMock()
    database.__getitem__.return_value = collection
    return DocumentRepository(database), collection


@pytest.mark.unit
def test_create_document_inserts_document_payload():
    repository, collection = create_repository()
    collection.insert_one = AsyncMock()
    document = create_document()

    result = asyncio.run(repository.create_document(document, "uploads/document-01/source.pdf"))

    assert result == document
    payload = collection.insert_one.call_args.args[0]
    assert payload["document_id"] == "document-01"
    assert payload["file_type"] == "pdf"
    assert payload["status"] == "pending"
    assert payload["storage_path"] == "uploads/document-01/source.pdf"
    assert isinstance(payload["updated_at"], datetime)


@pytest.mark.unit
def test_get_document_returns_document_when_found():
    repository, collection = create_repository()
    document = create_document()
    collection.find_one = AsyncMock(return_value=document.model_dump(mode="python"))

    result = asyncio.run(repository.get_document("document-01"))

    assert result == document
    collection.find_one.assert_called_once_with(
        {"document_id": "document-01"},
        {"_id": 0},
    )


@pytest.mark.unit
def test_get_document_returns_none_when_missing():
    repository, collection = create_repository()
    collection.find_one = AsyncMock(return_value=None)

    assert asyncio.run(repository.get_document("missing")) is None


@pytest.mark.unit
def test_update_document_status_updates_fields():
    repository, collection = create_repository()
    document = create_document()
    document.status = DocumentStatus.COMPLETED
    document.total_pages = 3
    collection.find_one_and_update = AsyncMock(return_value=document.model_dump(mode="python"))

    result = asyncio.run(
        repository.update_document_status(
            "document-01",
            DocumentStatus.COMPLETED,
            total_pages=3,
            error_message="",
        )
    )

    assert result == document
    update_data = collection.find_one_and_update.call_args.args[1]["$set"]
    assert update_data["status"] == "completed"
    assert update_data["total_pages"] == 3
    assert update_data["error_message"] == ""
    assert isinstance(update_data["updated_at"], datetime)


@pytest.mark.unit
def test_update_document_status_rejects_invalid_page_count():
    repository, collection = create_repository()
    collection.find_one_and_update = AsyncMock()

    with pytest.raises(ValueError, match="greater than zero"):
        asyncio.run(
            repository.update_document_status(
                "document-01",
                DocumentStatus.COMPLETED,
                total_pages=0,
            )
        )

    collection.find_one_and_update.assert_not_called()


@pytest.mark.unit
def test_update_document_status_returns_none_when_missing():
    repository, collection = create_repository()
    collection.find_one_and_update = AsyncMock(return_value=None)

    result = asyncio.run(repository.update_document_status("missing", DocumentStatus.FAILED))

    assert result is None


@pytest.mark.unit
def test_delete_document_returns_true_when_deleted():
    repository, collection = create_repository()
    collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))

    assert asyncio.run(repository.delete_document("document-01")) is True
    collection.delete_one.assert_called_once_with({"document_id": "document-01"})


@pytest.mark.unit
def test_delete_document_returns_false_when_missing():
    repository, collection = create_repository()
    collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=0))

    assert asyncio.run(repository.delete_document("missing")) is False
