
from typing import Annotated

from app.core.database import get_database
from app.features.document_extraction.repository import DocumentRepository
from app.features.document_extraction.schemas import (
    DocumentResponse,
    UploadResponse,
)
from app.features.document_extraction.service import DocumentService
from app.features.document_extraction.validators import DocumentValidationError
from app.storage.local import FileStorageError
from fastapi import (
    APIRouter,
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
        File(
            description=(
                "Upload PDF, JPG, PNG, TIFF, "
                "XLSX, XLS, or CSV documents."
            )
        ),
    ],
    service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],
) -> UploadResponse:

    try:
        return await service.upload_documents(files)

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
