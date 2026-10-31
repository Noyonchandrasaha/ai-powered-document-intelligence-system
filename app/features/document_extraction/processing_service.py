"""Background orchestration for document extraction jobs."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.features.document_extraction.data_processing.ocr.google_vision import (
    GoogleVisionOCR,
)
from app.features.document_extraction.data_processing.pipeline import (
    process_document,
)
from app.features.document_extraction.repository import (
    DocumentRepository,
    ProcessingRepository,
)
from app.features.document_extraction.schemas import (
    DocumentStatus,
)
from app.storage.local import get_file_path
from pymongo.asynchronous.database import AsyncDatabase

logger = get_logger(__name__)


class DocumentProcessingError(RuntimeError):
    """Raised when a document processing job cannot be completed."""


class DocumentProcessingService:
    """Run, persist and monitor document extraction jobs."""

    def __init__(self, database: AsyncDatabase[Any]) -> None:
        self.documents = DocumentRepository(database)
        self.processing = ProcessingRepository(database)

    async def process_document(
        self,
        document_id: str,
    ) -> DocumentStatus:
        started_at = datetime.now(UTC)
        document = await self.documents.get_document(document_id)

        if document is None:
            raise DocumentProcessingError("Document not found.")

        await self.documents.update_document_status(
            document_id,
            DocumentStatus.PROCESSING,
        )
        await self.processing.add_history({
            "document_id": document_id,
            "status": DocumentStatus.PROCESSING.value,
            "processed_units": 0,
            "started_at": started_at,
            "completed_at": None,
            "error_message": None,
        })

        try:
            storage_path = await self.documents.get_storage_path(document_id)
            file_path = (
                Path(storage_path)
                if storage_path
                else get_file_path(document_id)
            )

            ocr: GoogleVisionOCR | None = None
            credentials_file = Path(
                get_settings().google_vision_credentials_file
            )

            if credentials_file.is_file():
                ocr = GoogleVisionOCR(credentials_file)

            results: list[dict[str, Any]] = []

            if ocr is None:
                async for result in process_document(
                    document_id,
                    file_path,
                    document.file_type,
                ):
                    results.append(dict(result))
            else:
                async with ocr:
                    async for result in process_document(
                        document_id,
                        file_path,
                        document.file_type,
                        ocr,
                    ):
                        results.append(dict(result))

            await self.processing.replace_results(document_id, results)

            final_status = (
                DocumentStatus.COMPLETED
                if results
                and all(item["status"] == "completed" for item in results)
                else DocumentStatus.PARTIAL
                if results
                else DocumentStatus.FAILED
            )
            total_pages = self._total_pages(results)
            error_message = self._first_error(results)

            await self.documents.update_document_status(
                document_id,
                final_status,
                total_pages=total_pages,
                error_message=error_message,
            )
            await self.processing.add_history({
                "document_id": document_id,
                "status": final_status.value,
                "processed_units": len(results),
                "started_at": started_at,
                "completed_at": datetime.now(UTC),
                "error_message": error_message,
            })

            return final_status

        except Exception as exc:
            logger.exception(
                "document_processing_failed",
                document_id=document_id,
                error=str(exc),
            )
            message = f"{type(exc).__name__}: {exc}"
            await self.documents.update_document_status(
                document_id,
                DocumentStatus.FAILED,
                error_message=message,
            )
            await self.processing.add_history({
                "document_id": document_id,
                "status": DocumentStatus.FAILED.value,
                "processed_units": 0,
                "started_at": started_at,
                "completed_at": datetime.now(UTC),
                "error_message": message,
            })
            return DocumentStatus.FAILED

    @staticmethod
    def _total_pages(results: list[dict[str, Any]]) -> int | None:
        for result in results:
            total_pages = result.get("metadata", {}).get("total_pages")
            if isinstance(total_pages, int) and total_pages > 0:
                return total_pages
        return None

    @staticmethod
    def _first_error(results: list[dict[str, Any]]) -> str | None:
        for result in results:
            error = result.get("error")
            if isinstance(error, str) and error:
                return error
        return None


async def process_document_job(
    database: AsyncDatabase[Any],
    document_id: str,
) -> None:
    """BackgroundTasks entry point."""

    await DocumentProcessingService(database).process_document(document_id)
