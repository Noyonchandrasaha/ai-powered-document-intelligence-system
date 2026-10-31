from datetime import UTC, datetime
from typing import Any

from app.features.document_extraction.schemas import (
    DocumentResponse,
    DocumentStatus,
)
from pymongo import ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase


class DocumentRepository:
    def __init__(self, database: AsyncDatabase[Any]):
        self.collection = database["documents"]

    async def create_document(
        self,
        document: DocumentResponse,
        storage_path: str,
    ) -> DocumentResponse:

        payload = document.model_dump(mode="python")

        payload["file_type"] = document.file_type.value
        payload["status"] = document.status.value
        payload["storage_path"] = storage_path
        payload["updated_at"] = datetime.now(UTC)

        await self.collection.insert_one(payload)

        return document

    async def get_document(
        self,
        document_id: str,
    ) -> DocumentResponse | None:

        document = await self.collection.find_one(
            {"document_id": document_id},
            {"_id": 0},
        )

        if document is None:
            return None

        return DocumentResponse.model_validate(document)

    async def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        total_pages: int | None = None,
        error_message: str | None = None,
    ) -> DocumentResponse | None:

        update_data: dict[str, Any] = {
            "status": status.value,
            "updated_at": datetime.now(UTC),
        }

        if total_pages is not None:
            if total_pages < 1:
                raise ValueError(
                    "total_pages must be greater than zero."
                )

            update_data["total_pages"] = total_pages

        if error_message is not None:
            update_data["error_message"] = error_message

        document = await self.collection.find_one_and_update(
            {"document_id": document_id},
            {"$set": update_data},
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )

        if document is None:
            return None

        return DocumentResponse.model_validate(document)

    async def delete_document(
        self,
        document_id: str,
    ) -> bool:

        result = await self.collection.delete_one(
            {"document_id": document_id}
        )

        return result.deleted_count == 1
