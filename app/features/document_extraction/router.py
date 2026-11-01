from typing import Annotated

from app.core.database import get_database
from app.features.document_extraction.processing_service import (
    DocumentProcessingService,
)
from app.features.document_extraction.repository import (
    DocumentRepository,
    ProcessingRepository,
)
from app.features.document_extraction.schemas import (
    DocumentResponse,
    DocumentStatus,
    DocumentStatusResponse,
    ProcessingHistoryResponse,
    UploadResponse,
)
from app.features.document_extraction.service import DocumentService
from app.features.document_extraction.validators import DocumentValidationError
from app.storage.local import FileStorageError
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Path,
    UploadFile,
    status,
)
from pymongo.asynchronous.database import AsyncDatabase

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


def get_document_service(
    database: Annotated[
        AsyncDatabase[dict[str, object]],
        Depends(get_database),
    ],
) -> DocumentService:

    repository = DocumentRepository(database)

    return DocumentService(repository)


def get_processing_service(
    database: Annotated[
        AsyncDatabase[dict[str, object]],
        Depends(get_database),
    ],
) -> DocumentProcessingService:
    return DocumentProcessingService(database)


def get_processing_repository(
    database: Annotated[
        AsyncDatabase[dict[str, object]],
        Depends(get_database),
    ],
) -> ProcessingRepository:
    return ProcessingRepository(database)


DocumentId = Annotated[
    str,
    Path(
        pattern=r"^[0-9a-f]{32}$",
        description="Unique document identifier",
    ),
]


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload multiple documents",
)
async def upload_documents(
    files: Annotated[
        list[UploadFile],
        File(description=("Upload PDF, JPG, PNG, TIFF, XLSX, XLS, or CSV documents.")),
    ],
    service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],
    background_tasks: BackgroundTasks,
    processing_service: Annotated[
        DocumentProcessingService,
        Depends(get_processing_service),
    ],
) -> UploadResponse:

    try:
        response = await service.upload_documents(files)
        for document in response.documents:
            await processing_service.enqueue_document(document.document_id)
        return response

    except (
        DocumentValidationError,
        FileStorageError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get(
    "/search",
    response_model=list[DocumentResponse],
    summary="Search documents",
)
async def search_documents(
    database: Annotated[
        AsyncDatabase[dict[str, object]],
        Depends(get_database),
    ],
    query: str | None = None,
    document_status: DocumentStatus | None = None,
    limit: int = 50,
) -> list[DocumentResponse]:
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="limit must be between 1 and 100.",
        )

    return await DocumentRepository(database).search_documents(
        query=query,
        status=document_status,
        limit=limit,
    )


@router.get("/{document_id}/units", summary="Inspect source pages, sheets and rows")
async def source_units(
    document_id: DocumentId,
    database: Annotated[AsyncDatabase[dict[str, object]], Depends(get_database)],
    offset: int = 0,
    limit: int = 50,
) -> list[dict[str, object]]:
    if offset < 0 or not 1 <= limit <= 100:
        raise HTTPException(422, "offset must be nonnegative and limit must be 1..100.")
    if await DocumentRepository(database).get_document(document_id) is None:
        raise HTTPException(404, "Document not found.")
    return (
        await database["extraction_results"]
        .find(
            {"document_id": document_id},
            {"_id": 0},
        )
        .sort([("page_number", 1), ("sheet_index", 1), ("start_row", 1)])
        .skip(offset)
        .limit(limit)
        .to_list(length=limit)
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document metadata",
)
async def get_document(
    document_id: DocumentId,
    service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],
) -> DocumentResponse:

    document = await service.get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return document


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
)
async def delete_document(
    document_id: DocumentId,
    service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],
) -> None:

    deleted = await service.delete_document(document_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get document processing status",
)
async def get_document_status(
    document_id: DocumentId,
    database: Annotated[
        AsyncDatabase[dict[str, object]],
        Depends(get_database),
    ],
) -> DocumentStatusResponse:
    status_response = await DocumentRepository(database).get_document_status(document_id)

    if status_response is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return status_response


@router.post(
    "/{document_id}/reprocess",
    response_model=DocumentStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Reprocess a document",
)
async def reprocess_document(
    document_id: DocumentId,
    background_tasks: BackgroundTasks,
    database: Annotated[
        AsyncDatabase[dict[str, object]],
        Depends(get_database),
    ],
    processing_service: Annotated[
        DocumentProcessingService,
        Depends(get_processing_service),
    ],
) -> DocumentStatusResponse:
    repository = DocumentRepository(database)
    document = await repository.get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    if document.status not in {
        DocumentStatus.FAILED,
        DocumentStatus.PARTIAL,
    }:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=("Only failed or partial documents can be reprocessed."),
        )

    await repository.update_document_status(
        document_id,
        DocumentStatus.PENDING,
    )
    await processing_service.enqueue_document(document_id)

    return DocumentStatusResponse(
        document_id=document_id,
        status=DocumentStatus.PENDING,
    )


@router.get(
    "/{document_id}/history",
    response_model=list[ProcessingHistoryResponse],
    summary="Get document processing history",
)
async def get_processing_history(
    document_id: DocumentId,
    database: Annotated[
        AsyncDatabase[dict[str, object]],
        Depends(get_database),
    ],
    processing_repository: Annotated[
        ProcessingRepository,
        Depends(get_processing_repository),
    ],
) -> list[ProcessingHistoryResponse]:
    if await DocumentRepository(database).get_document(document_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    history = await processing_repository.get_history(document_id)
    return [ProcessingHistoryResponse.model_validate(item) for item in history]


@router.get("/{document_id}/candidates", summary="List automatic invoice groups")
async def invoice_candidates(
    document_id: DocumentId,
    database: Annotated[AsyncDatabase[dict[str, object]], Depends(get_database)],
    offset: int = 0,
    limit: int = 50,
) -> list[dict[str, object]]:
    if offset < 0 or not 1 <= limit <= 100:
        raise HTTPException(422, "Invalid pagination.")
    return (
        await database["invoice_candidates"]
        .find(
            {"document_id": document_id},
            {"_id": 0},
        )
        .sort("candidate_id", 1)
        .skip(offset)
        .limit(limit)
        .to_list(length=limit)
    )
