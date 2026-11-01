"""FastAPI routes for AI invoice information extraction."""

from typing import Annotated, Any

from app.core.config import get_settings
from app.core.database import get_database
from app.features.document_extraction.repository import (
    DocumentRepository,
)
from app.features.document_extraction.schemas import (
    DocumentStatus,
)
from app.features.invoices.context import (
    MAX_SOURCE_UNITS,
)
from app.features.invoices.extractor import (
    InvoiceExtractionError,
    OpenAIInvoiceExtractor,
)
from app.features.invoices.repository import (
    InvoiceRepository,
)
from app.features.invoices.schemas import (
    ExtractionStatus,
    InvoiceData,
    InvoiceExtractionResponse,
)
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    status,
)
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field
from pymongo.asynchronous.database import AsyncDatabase

router = APIRouter(
    prefix="/invoices",
    tags=["Invoice Extraction"],
)


DocumentId = Annotated[
    str,
    Path(
        pattern=r"^[0-9a-f]{32}$",
        description="Document UUID in hexadecimal format",
    ),
]

ExtractionId = Annotated[
    str,
    Path(
        pattern=r"^[0-9a-f]{32}$",
        description="Invoice extraction UUID in hexadecimal format",
    ),
]


class ExtractInvoiceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_SOURCE_UNITS,
        description=(
            "Optional Task 1 source unit IDs. "
            "Required when the document has too many "
            "pages or contains multiple invoices."
        ),
    )

    extraction_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{32}$",
        description=("Optional existing extraction ID for idempotent retries."),
    )


class ExtractionRunResponse(BaseModel):
    extraction_id: str
    result: InvoiceExtractionResponse


class InvoiceCorrectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice: InvoiceData
    corrected_by: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=1000)


def get_invoice_repository(
    database: Annotated[
        AsyncDatabase[Any],
        Depends(get_database),
    ],
) -> InvoiceRepository:
    return InvoiceRepository(database)


async def load_document_results(
    database: AsyncDatabase[Any],
    document_id: str,
    unit_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Read trusted Task 1 results from MongoDB.

    Assumes the collection is named extraction_results
    and stores normalized ProcessingResult records.

    This read adapter can be moved to the Task 1
    repository once its concrete schema is confirmed.
    """

    if unit_ids is not None:
        if len(set(unit_ids)) != len(unit_ids):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Duplicate source unit IDs.",
            )

    query: dict[str, Any] = {
        "document_id": document_id,
    }

    if unit_ids is not None:
        query["unit_id"] = {
            "$in": unit_ids,
        }

    cursor = (
        database["extraction_results"]
        .find(
            query,
            {"_id": 0},
        )
        .limit(MAX_SOURCE_UNITS + 1)
    )

    results = await cursor.to_list(length=MAX_SOURCE_UNITS + 1)

    if len(results) > MAX_SOURCE_UNITS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Document exceeds invoice context limits. Select the relevant invoice source units."
            ),
        )

    if unit_ids is not None:
        found = {item.get("unit_id") for item in results}

        if found != set(unit_ids):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=("One or more requested source units were not found."),
            )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=("No Task 1 extraction results found for this document."),
        )

    return results


@router.post(
    "/documents/{document_id}/extract",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue structured invoice extraction",
)
async def extract_invoice(
    document_id: DocumentId,
    payload: ExtractInvoiceRequest,
    database: Annotated[AsyncDatabase[Any], Depends(get_database)],
) -> dict[str, Any]:
    from app.features.jobs import JobQueue

    document = await DocumentRepository(database).get_document(document_id)
    if document is None:
        raise HTTPException(404, "Document not found.")
    if document.status not in {DocumentStatus.COMPLETED, DocumentStatus.PARTIAL}:
        raise HTTPException(409, "Document processing must finish before invoice extraction.")
    if not get_settings().openai_api_key:
        raise HTTPException(503, "OpenAI integration is not configured.")
    if payload.unit_ids:
        await load_document_results(database, document_id, payload.unit_ids)
    return await JobQueue(database).enqueue(
        "extract",
        document_id,
        payload.model_dump(mode="json"),
    )


@router.get(
    "/extractions/{extraction_id}",
    response_model=InvoiceExtractionResponse,
    summary="Get invoice extraction result",
)
async def get_invoice_extraction(
    extraction_id: ExtractionId,
    repository: Annotated[
        InvoiceRepository,
        Depends(get_invoice_repository),
    ],
) -> InvoiceExtractionResponse:

    result = await repository.get_extraction(extraction_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invoice extraction not found.",
        )

    return result


@router.patch(
    "/extractions/{extraction_id}",
    response_model=InvoiceExtractionResponse,
    summary="Apply a manual invoice correction",
)
async def correct_invoice_extraction(
    extraction_id: ExtractionId,
    payload: InvoiceCorrectionRequest,
    repository: Annotated[
        InvoiceRepository,
        Depends(get_invoice_repository),
    ],
) -> InvoiceExtractionResponse:
    result = await repository.correct_extraction(
        extraction_id,
        payload.invoice,
        corrected_by=payload.corrected_by,
        reason=payload.reason,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invoice extraction not found.",
        )

    return result


@router.get(
    "/search",
    response_model=list[InvoiceExtractionResponse],
    summary="Search invoice extractions",
)
async def search_invoice_extractions(
    repository: Annotated[
        InvoiceRepository,
        Depends(get_invoice_repository),
    ],
    query: str | None = None,
    extraction_status: ExtractionStatus | None = None,
    limit: int = 50,
) -> list[InvoiceExtractionResponse]:
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="limit must be between 1 and 100.",
        )

    return await repository.search_extractions(
        query=query,
        status=extraction_status,
        limit=limit,
    )


@router.get(
    "/search/semantic",
    response_model=list[InvoiceExtractionResponse],
    summary="Semantic-search invoice extractions",
)
async def semantic_search_invoice_extractions(
    query: str,
    database: Annotated[
        AsyncDatabase[Any],
        Depends(get_database),
    ],
    status_filter: ExtractionStatus | None = None,
    limit: int = 50,
) -> list[InvoiceExtractionResponse]:
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="query cannot be empty.",
        )
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="limit must be between 1 and 100.",
        )

    settings = get_settings()
    if not settings.openai_api_key or not settings.semantic_search_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Semantic search is not configured.",
        )

    try:
        async with AsyncOpenAI(api_key=settings.openai_api_key) as client:
            extractor = OpenAIInvoiceExtractor(
                client=client,
                model=settings.openai_model,
                embedding_model=settings.openai_embedding_model,
            )
            query_vector = await extractor.embed(query)
    except InvoiceExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return await InvoiceRepository(database).semantic_search(
        query_vector,
        status=status_filter,
        limit=limit,
    )


@router.get(
    "/documents/{document_id}/latest",
    response_model=InvoiceExtractionResponse,
    summary="Get latest invoice extraction",
)
async def get_latest_invoice_extraction(
    document_id: DocumentId,
    repository: Annotated[
        InvoiceRepository,
        Depends(get_invoice_repository),
    ],
) -> InvoiceExtractionResponse:

    result = await repository.get_latest_extraction(document_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No invoice extraction found.",
        )

    return result
