import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

MAX_CONTEXT_CHARS = 100_000
MAX_SOURCE_UNITS = 100
MAX_SOURCE_RECORDS = 1000


class InvoiceContextError(ValueError):
    """Raised when invoice context preparation fails."""


class ContextLimitExceeded(InvoiceContextError):
    """Raised when source content exceeds context limits."""


@dataclass(frozen=True, slots=True)
class InvoiceContext:
    document_id: str
    content: str
    source_units: tuple[str, ...]
    source_record_count: int
    incomplete: bool
    warnings: tuple[str, ...]


def _iter_page_sources(
    result: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    """Preserve native and OCR text as separate sources."""

    unit_id = result["unit_id"]
    page_number = result.get("page_number")

    native_text = result.get("native_text") or ""
    ocr_text = result.get("ocr_text") or ""
    fallback_text = result.get("text") or ""

    if native_text.strip():
        yield {
            "source_id": f"{unit_id}:native",
            "kind": "page",
            "page_number": page_number,
            "method": "native_text",
            "text": native_text,
        }

    if ocr_text.strip():
        yield {
            "source_id": f"{unit_id}:ocr",
            "kind": "page",
            "page_number": page_number,
            "method": "ocr",
            "text": ocr_text,
        }

    if not native_text.strip() and not ocr_text.strip() and fallback_text.strip():
        yield {
            "source_id": f"{unit_id}:text",
            "kind": "page",
            "page_number": page_number,
            "method": result.get("extraction_method"),
            "text": fallback_text,
        }


def _iter_tabular_sources(
    result: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    """Preserve original spreadsheet rows and cell references."""

    unit_id = result["unit_id"]

    for row in result.get("rows", []):
        cells = []

        for cell in row.get("cells", []):
            if cell.get("value") is None and not cell.get("formula"):
                continue

            cells.append(
                {
                    "coordinate": cell.get("coordinate"),
                    "value": cell.get("value"),
                    "formula": cell.get("formula"),
                    "data_type": cell.get("data_type"),
                }
            )

        if not cells:
            continue

        row_number = row.get("row_number")

        yield {
            "source_id": f"{unit_id}:row:{row_number}",
            "kind": result["kind"],
            "sheet_name": result.get("sheet_name"),
            "sheet_index": result.get("sheet_index"),
            "row_number": row_number,
            "line_number": row.get("line_number"),
            "cells": cells,
        }


def _iter_source_records(
    result: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    """Convert a Task 1 processing result into AI-ready records."""

    kind = result.get("kind")

    if kind == "page":
        yield from _iter_page_sources(result)

    elif kind in {"sheet_chunk", "csv_chunk"}:
        yield from _iter_tabular_sources(result)

    else:
        raise InvoiceContextError(f"Unsupported processing result kind: {kind}")


def build_invoice_context(
    document_id: str,
    results: Iterable[Mapping[str, Any]],
    *,
    max_chars: int = MAX_CONTEXT_CHARS,
    max_units: int = MAX_SOURCE_UNITS,
    max_records: int = MAX_SOURCE_RECORDS,
) -> InvoiceContext:
    """
    Build a bounded, source-referenced context for ONE invoice.

    The caller must select the relevant document units first.

    This function:
    - Does not call an AI model.
    - Does not access MongoDB.
    - Does not infer missing financial values.
    - Does not silently truncate source content.
    """

    if not document_id.strip():
        raise InvoiceContextError("Document ID is required.")

    if min(max_chars, max_units, max_records) <= 0:
        raise ValueError("Context limits must be positive.")

    header = f"DOCUMENT_ID: {document_id}\nSOURCE_RECORDS_JSONL:\n"

    content_parts = [header]
    current_size = len(header)

    included_units: list[str] = []
    warnings: list[str] = []

    unit_count = 0
    record_count = 0
    incomplete = False

    for result in results:
        unit_count += 1

        if unit_count > max_units:
            raise ContextLimitExceeded(
                "Too many source units. "
                "Select the relevant invoice pages "
                "or spreadsheet ranges first."
            )

        if result.get("document_id") != document_id:
            raise InvoiceContextError("Source document ID does not match.")

        unit_id = result.get("unit_id")

        if not isinstance(unit_id, str) or not unit_id:
            raise InvoiceContextError("Processing result has no valid unit ID.")

        status = result.get("status")

        if status not in {"completed", "partial", "failed"}:
            raise InvoiceContextError(f"Invalid source status: {status}")

        if status in {"partial", "failed"}:
            incomplete = True

            warnings.append(
                f"{unit_id}: {status} ({result.get('error') or 'incomplete extraction'})"
            )

        unit_has_content = False

        for record in _iter_source_records(result):
            record_count += 1

            if record_count > max_records:
                raise ContextLimitExceeded(
                    "Too many source records. "
                    "Split the invoice context into "
                    "smaller relevant sections."
                )

            serialized = (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
                + "\n"
            )

            if current_size + len(serialized) > max_chars:
                raise ContextLimitExceeded(
                    "Invoice context exceeds the "
                    "configured character limit. "
                    "Prepare smaller invoice-specific "
                    "page or row groups."
                )

            content_parts.append(serialized)
            current_size += len(serialized)

            unit_has_content = True

        if unit_has_content:
            included_units.append(unit_id)

    if record_count == 0:
        raise InvoiceContextError("No extractable invoice content was found.")

    return InvoiceContext(
        document_id=document_id,
        content="".join(content_parts),
        source_units=tuple(included_units),
        source_record_count=record_count,
        incomplete=incomplete,
        warnings=tuple(warnings),
    )
