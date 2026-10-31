import re
from datetime import UTC, datetime
from typing import Any

from app.features.document_extraction.schemas import (
    DocumentResponse,
    DocumentStatus,
    DocumentStatusResponse,
)
from pymongo import ASCENDING, ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase


class DocumentRepository:
    def __init__(self, database: AsyncDatabase[Any]):
        self.collection = database["documents"]

    async def ensure_indexes(self) -> None:
        await self.collection.create_index(
            [("document_id", ASCENDING)],
            unique=True,
            name="uq_document_id",
        )
        await self.collection.create_index(
            [("status", ASCENDING), ("created_at", -1)],
            name="idx_document_status_created",
        )

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

    async def get_storage_path(
        self,
        document_id: str,
    ) -> str | None:
        document = await self.collection.find_one(
            {"document_id": document_id},
            {"_id": 0, "storage_path": 1},
        )

        if document is None:
            return None

        storage_path = document.get("storage_path")
        return storage_path if isinstance(storage_path, str) else None

    async def search_documents(
        self,
        *,
        query: str | None = None,
        status: DocumentStatus | None = None,
        limit: int = 50,
    ) -> list[DocumentResponse]:
        filters: dict[str, Any] = {}

        if query:
            safe_query = re.escape(query.strip())
            filters["$or"] = [
                {"document_id": {"$regex": safe_query, "$options": "i"}},
                {"filename": {"$regex": safe_query, "$options": "i"}},
            ]

        if status is not None:
            filters["status"] = status.value

        cursor = self.collection.find(
            filters,
            {"_id": 0},
        ).sort("created_at", -1).limit(limit)

        documents = await cursor.to_list(length=limit)
        return [DocumentResponse.model_validate(item) for item in documents]

    async def get_document_status(
        self,
        document_id: str,
    ) -> DocumentStatusResponse | None:
        document = await self.collection.find_one(
            {"document_id": document_id},
            {
                "_id": 0,
                "document_id": 1,
                "status": 1,
                "total_pages": 1,
                "error_message": 1,
                "updated_at": 1,
            },
        )

        if document is None:
            return None

        return DocumentStatusResponse.model_validate(document)


class ProcessingRepository:
    """Persistence adapter for extraction units and processing history."""

    def __init__(self, database: AsyncDatabase[Any]) -> None:
        self.results = database["extraction_results"]
        self.history = database["processing_history"]

    async def replace_results(
        self,
        document_id: str,
        results: list[dict[str, Any]],
    ) -> None:
        await self.results.delete_many({"document_id": document_id})
        if results:
            await self.results.insert_many(results, ordered=True)

    async def get_results(
        self,
        document_id: str,
        unit_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"document_id": document_id}
        if unit_ids is not None:
            query["unit_id"] = {"$in": unit_ids}

        cursor = self.results.find(
            query,
            {"_id": 0},
        ).sort("unit_id", 1)
        return await cursor.to_list(length=10_000)

    async def add_history(self, record: dict[str, Any]) -> None:
        await self.history.insert_one(record)

    async def get_history(
        self,
        document_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        cursor = self.history.find(
            {"document_id": document_id},
            {"_id": 0},
        ).sort("started_at", -1).limit(limit)
        return await cursor.to_list(length=limit)

    async def ensure_indexes(self) -> None:
        await self.results.create_index(
            [("document_id", 1), ("unit_id", 1)],
            unique=True,
            name="uq_document_unit",
        )
        await self.history.create_index(
            [("document_id", 1), ("started_at", -1)],
            name="idx_processing_history",
        )
