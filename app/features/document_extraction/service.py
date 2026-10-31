
import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.core.logging import get_logger
from app.features.document_extraction.repository import DocumentRepository
from app.features.document_extraction.schemas import (
    DocumentResponse,
    DocumentStatus,
    UploadResponse,
)
from app.features.document_extraction.validators import (
    validate_document,
    validate_extension,
)
from app.storage.local import delete_file, save_file
from fastapi import UploadFile

logger = get_logger(__name__)

MAX_FILES_PER_BATCH = 20


class DocumentService:

    def __init__(
        self,
        repository: DocumentRepository,
    ) -> None:
        self.repository = repository

    async def upload_document(
        self,
        file: UploadFile,
    ) -> DocumentResponse:

        # Validate the filename and extension
        document_type = validate_extension(file.filename)

        original_filename = Path(
            (file.filename or "").replace("\\", "/")
        ).name

        document_id = uuid4().hex
        file_saved = False

        try:
            # Save the uploaded file locally
            file_path, file_size = await save_file(
                file=file,
                document_id=document_id,
                extension=document_type.value,
            )

            file_saved = True

            # Validate signature and file integrity
            validated_type, validated_size = (
                await asyncio.to_thread(
                    validate_document,
                    file_path,
                    original_filename,
                )
            )

            # Construct document metadata
            document = DocumentResponse(
                document_id=document_id,
                filename=original_filename,
                file_type=validated_type,
                file_size=validated_size,
                status=DocumentStatus.PENDING,
                created_at=datetime.now(UTC),
                total_pages=None,
            )

            # Persist metadata in MongoDB
            await self.repository.create_document(
                document=document,
                storage_path=str(file_path),
            )

            logger.info(
                "document_registered",
                document_id=document_id,
            )

            return document

        except Exception:
            logger.exception(
                "document_upload_failed",
                document_id=document_id,
            )

            if file_saved:
                try:
                    await asyncio.to_thread(
                        delete_file,
                        document_id,
                    )
                except Exception:
                    logger.exception(
                        "document_upload_cleanup_failed",
                        document_id=document_id,
                    )

            raise

    async def upload_documents(
        self,
        files: list[UploadFile],
    ) -> UploadResponse:

        if not files:
            raise ValueError(
                "At least one file is required."
            )

        if len(files) > MAX_FILES_PER_BATCH:
            raise ValueError(
                f"Maximum {MAX_FILES_PER_BATCH} files allowed."
            )

        batch_id = uuid4().hex
        uploaded_documents: list[DocumentResponse] = []

        try:
            for file in files:
                document = await self.upload_document(file)

                uploaded_documents.append(document)

        except Exception:
            logger.exception(
                "document_batch_upload_failed",
                batch_id=batch_id,
            )

            # Best-effort rollback of previous uploads
            for document in reversed(uploaded_documents):
                try:
                    await self.delete_document(
                        document.document_id
                    )
                except Exception:
                    logger.exception(
                        "document_batch_rollback_failed",
                        document_id=document.document_id,
                    )

            raise

        return UploadResponse(
            batch_id=batch_id,
            total_files=len(uploaded_documents),
            documents=uploaded_documents,
        )

    async def get_document(
        self,
        document_id: str,
    ) -> DocumentResponse | None:

        return await self.repository.get_document(
            document_id
        )

    async def delete_document(
        self,
        document_id: str,
    ) -> bool:

        document = await self.repository.get_document(
            document_id
        )

        if document is None:
            return False

        deleted = await self.repository.delete_document(
            document_id
        )

        if deleted:
            await asyncio.to_thread(
                delete_file,
                document_id,
            )

            logger.info(
                "document_deleted",
                document_id=document_id,
            )

        return deleted
