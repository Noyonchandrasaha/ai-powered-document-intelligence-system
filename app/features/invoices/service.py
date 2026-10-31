
"""Application service for AI invoice extraction."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from app.core.logging import get_logger
from app.features.invoices.context import (
    MAX_SOURCE_UNITS,
    InvoiceContext,
    build_invoice_context,
)
from app.features.invoices.extractor import (
    InvoiceExtractionError,
    OpenAIInvoiceExtractor,
)
from app.features.invoices.prompts import (
    INVOICE_PROMPT_VERSION,
)
from app.features.invoices.repository import (
    InvoiceRepository,
)
from app.features.invoices.schemas import (
    ExtractionStatus,
    InvoiceExtractionResponse,
)
from app.features.invoices.validators import (
    validate_invoice_extraction,
)

logger = get_logger(__name__)


class InvoiceServiceError(Exception):
    """Raised when the invoice workflow cannot proceed."""


@dataclass(frozen=True, slots=True)
class InvoiceExtractionOutcome:
    extraction_id: str
    result: InvoiceExtractionResponse


def _source_sort_key(
    result: Mapping[str, Any],
) -> tuple[int, int, int]:
    """Maintain natural page, sheet and CSV chunk order."""

    kind = result.get("kind")

    if kind == "page":
        return (
            0,
            int(result.get("page_number") or 0),
            0,
        )

    if kind == "sheet_chunk":
        return (
            1,
            int(result.get("sheet_index") or 0),
            int(result.get("start_row") or 0),
        )

    if kind == "csv_chunk":
        unit_id = str(result.get("unit_id") or "")

        match = re.fullmatch(
            r"csv:chunk:(\d+)",
            unit_id,
        )

        return (
            2,
            int(match.group(1)) if match else 0,
            0,
        )

    return (3, 0, 0)


class InvoiceService:
    """
    Coordinate the invoice extraction workflow.

    Responsibilities:
    - Validate supplied document results
    - Build source-referenced AI context
    - Execute OpenAI extraction
    - Validate extracted information
    - Persist validated invoice results
    """

    def __init__(
        self,
        repository: InvoiceRepository,
        extractor: OpenAIInvoiceExtractor,
        semantic_search_enabled: bool = False,
    ) -> None:
        self.repository = repository
        self.extractor = extractor
        self.semantic_search_enabled = semantic_search_enabled

    @staticmethod
    def _embedding_text(result: InvoiceExtractionResponse) -> str:
        parts: list[str] = []
        for field, value in result.invoice.model_dump(mode="json").items():
            if field == "line_items":
                for item in value:
                    parts.extend(
                        f"line_item_{key}: {item_value}"
                        for key, item_value in item.items()
                        if item_value is not None
                    )
            elif value is not None:
                parts.append(f"{field}: {value}")
        return "\n".join(parts)

    def _prepare_context(
        self,
        document_id: str,
        source_results: Sequence[Mapping[str, Any]],
        *,
        source_incomplete: bool = False,
    ) -> InvoiceContext:
        """
        Prepare context from Task 1 results.

        The supplied source units must belong to
        one invoice candidate.
        """

        if not document_id.strip():
            raise InvoiceServiceError(
                "Document ID is required."
            )

        if not source_results:
            raise InvoiceServiceError(
                "No document extraction results found."
            )

        if len(source_results) > MAX_SOURCE_UNITS:
            raise InvoiceServiceError(
                "Too many source units. "
                "Select invoice-specific pages or "
                "spreadsheet ranges before extraction."
            )

        seen_units: set[str] = set()

        for result in source_results:
            if result.get("document_id") != document_id:
                raise InvoiceServiceError(
                    "Source belongs to another document."
                )

            unit_id = result.get("unit_id")

            if not isinstance(unit_id, str) or not unit_id:
                raise InvoiceServiceError(
                    "Invalid source unit identifier."
                )

            if unit_id in seen_units:
                raise InvoiceServiceError(
                    f"Duplicate source unit: {unit_id}"
                )

            seen_units.add(unit_id)

        ordered_results = sorted(
            source_results,
            key=_source_sort_key,
        )

        context = build_invoice_context(
            document_id=document_id,
            results=ordered_results,
        )

        if source_incomplete and not context.incomplete:
            context = replace(
                context,
                incomplete=True,
                warnings=(
                    *context.warnings,
                    "Original document processing "
                    "was incomplete.",
                ),
            )

        return context

    async def extract_invoice(
        self,
        *,
        document_id: str,
        source_results: Sequence[Mapping[str, Any]],
        source_incomplete: bool = False,
        extraction_id: str | None = None,
    ) -> InvoiceExtractionOutcome:
        """
        Execute invoice extraction and save the result.

        source_results must be retrieved from the
        trusted Task 1 result storage.

        For a document containing multiple invoices,
        supply only the relevant invoice source units.
        """

        if extraction_id is not None:
            if (
                not extraction_id.strip()
                or len(extraction_id) > 128
            ):
                raise InvoiceServiceError(
                    "Invalid extraction ID."
                )

            # Return the previously stored result
            # when retrying a completed extraction.
            existing = await self.repository.get_extraction(
                extraction_id
            )

            if existing is not None:
                if existing.document_id != document_id:
                    raise InvoiceServiceError(
                        "Extraction ID belongs to "
                        "another document."
                    )

                return InvoiceExtractionOutcome(
                    extraction_id=extraction_id,
                    result=existing,
                )

        else:
            extraction_id = uuid4().hex

        # Step 1: Build AI-ready context.
        context = self._prepare_context(
            document_id=document_id,
            source_results=source_results,
            source_incomplete=source_incomplete,
        )

        logger.info(
            "invoice_extraction_started",
            document_id=document_id,
            extraction_id=extraction_id,
            units=len(context.source_units),
        )

        # Step 2: Extract structured data with OpenAI.
        ai_result = await self.extractor.extract(
            context
        )

        # Step 3: Validate against original source.
        validated_result = validate_invoice_extraction(
            document_id=document_id,
            ai_result=ai_result,
            context=context,
        )

        # An incomplete source cannot establish
        # with certainty that a document is not an invoice.
        if (
            context.incomplete
            and validated_result.status
            == ExtractionStatus.NOT_INVOICE
        ):
            validated_result = validated_result.model_copy(
                update={
                    "status": ExtractionStatus.NEEDS_REVIEW,
                    "uncertain_fields": ["is_invoice"],
                }
            )

        duplicate = await self.repository.find_duplicate_invoice(
            document_id,
            validated_result.invoice.invoice_number,
        )

        if duplicate is not None:
            duplicate_message = (
                "Invoice number already exists for another document."
            )
            uncertain_fields = list(validated_result.uncertain_fields)
            if "invoice_number" not in uncertain_fields:
                uncertain_fields.append("invoice_number")

            validation_errors = list(validated_result.validation_errors)
            if duplicate_message not in validation_errors:
                validation_errors.append(duplicate_message)

            validated_result = validated_result.model_copy(
                update={
                    "status": ExtractionStatus.NEEDS_REVIEW,
                    "uncertain_fields": uncertain_fields,
                    "validation_errors": validation_errors,
                }
            )

        # Step 4: Persist the validated result.
        embedding: list[float] | None = None
        if self.semantic_search_enabled:
            embedding_text = self._embedding_text(validated_result)
            if embedding_text:
                try:
                    embedding = await self.extractor.embed(embedding_text)
                except InvoiceExtractionError as exc:
                    logger.warning(
                        "invoice_embedding_failed",
                        document_id=document_id,
                        extraction_id=extraction_id,
                        error=str(exc),
                    )

        await self.repository.save_extraction(
            extraction_id=extraction_id,
            result=validated_result,
            model_name=self.extractor.model,
            prompt_version=INVOICE_PROMPT_VERSION,
            source_units=context.source_units,
            embedding=embedding,
        )

        logger.info(
            "invoice_extraction_saved",
            document_id=document_id,
            extraction_id=extraction_id,
            status=validated_result.status.value,
        )

        return InvoiceExtractionOutcome(
            extraction_id=extraction_id,
            result=validated_result,
        )

    async def get_extraction(
        self,
        extraction_id: str,
    ) -> InvoiceExtractionResponse | None:
        """Retrieve a previously saved extraction."""

        if not extraction_id.strip():
            raise InvoiceServiceError(
                "Extraction ID is required."
            )

        return await self.repository.get_extraction(
            extraction_id
        )

    async def get_latest_extraction(
        self,
        document_id: str,
    ) -> InvoiceExtractionResponse | None:
        """Retrieve the latest extraction for a document."""

        if not document_id.strip():
            raise InvoiceServiceError(
                "Document ID is required."
            )

        return await self.repository.get_latest_extraction(
            document_id
        )
