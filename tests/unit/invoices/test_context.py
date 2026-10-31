import pytest
from app.features.invoices.context import (
    ContextLimitExceeded,
    InvoiceContextError,
    build_invoice_context,
)


def _page_result(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "document_id": "doc-1",
        "unit_id": "page:1",
        "kind": "page",
        "status": "completed",
        "page_number": 1,
        "native_text": "Invoice total: 100.00",
        "ocr_text": "",
        "text": "Invoice total: 100.00",
        "extraction_method": "pymupdf",
    }
    result.update(overrides)
    return result


@pytest.mark.unit
def test_build_invoice_context_preserves_native_and_ocr_sources() -> None:
    result = build_invoice_context(
        "doc-1",
        [
            _page_result(),
            _page_result(
                unit_id="page:2",
                page_number=2,
                native_text="",
                ocr_text="Total: 100.00",
                text="Total: 100.00",
                extraction_method="google_vision",
            ),
        ],
    )

    assert result.document_id == "doc-1"
    assert result.source_units == ("page:1", "page:2")
    assert result.source_record_count == 2
    assert result.incomplete is False
    assert '"source_id":"page:1:native"' in result.content
    assert '"source_id":"page:2:ocr"' in result.content


@pytest.mark.unit
def test_build_invoice_context_uses_fallback_page_text() -> None:
    result = build_invoice_context(
        "doc-1",
        [
            _page_result(
                native_text="",
                ocr_text="",
                text="Fallback invoice text",
                extraction_method="custom",
            )
        ],
    )

    assert '"source_id":"page:1:text"' in result.content
    assert '"method":"custom"' in result.content


@pytest.mark.unit
def test_build_invoice_context_preserves_non_empty_tabular_cells() -> None:
    result = build_invoice_context(
        "doc-1",
        [
            {
                "document_id": "doc-1",
                "unit_id": "sheet:1:row:1",
                "kind": "sheet_chunk",
                "status": "completed",
                "rows": [
                    {
                        "row_number": 1,
                        "cells": [
                            {
                                "coordinate": "A1",
                                "value": "Total",
                                "formula": None,
                            },
                            {
                                "coordinate": "B1",
                                "value": None,
                                "formula": "=SUM(B2:B3)",
                            },
                            {
                                "coordinate": "C1",
                                "value": None,
                                "formula": None,
                            },
                        ],
                    }
                ],
            }
        ],
    )

    assert result.source_units == ("sheet:1:row:1",)
    assert '"coordinate":"A1"' in result.content
    assert '"coordinate":"B1"' in result.content
    assert '"coordinate":"C1"' not in result.content


@pytest.mark.unit
def test_build_invoice_context_marks_partial_sources_incomplete() -> None:
    result = build_invoice_context(
        "doc-1",
        [
            _page_result(
                status="partial",
                error="OCR unavailable",
            )
        ],
    )

    assert result.incomplete is True
    assert result.warnings == ("page:1: partial (OCR unavailable)",)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("results", "message"),
    [
        ([_page_result(document_id="other")], "Source document ID does not match."),
        ([_page_result(status="unknown")], "Invalid source status: unknown"),
        ([_page_result(kind="unknown")], "Unsupported processing result kind: unknown"),
    ],
)
def test_build_invoice_context_rejects_invalid_sources(
    results: list[dict[str, object]],
    message: str,
) -> None:
    with pytest.raises(InvoiceContextError, match=message):
        build_invoice_context("doc-1", results)


@pytest.mark.unit
def test_build_invoice_context_rejects_missing_content() -> None:
    with pytest.raises(
        InvoiceContextError,
        match="No extractable invoice content was found.",
    ):
        build_invoice_context(
            "doc-1",
            [_page_result(native_text="", text="")],
        )


@pytest.mark.unit
def test_build_invoice_context_rejects_invalid_arguments() -> None:
    with pytest.raises(InvoiceContextError, match="Document ID is required."):
        build_invoice_context("  ", [])

    with pytest.raises(ValueError, match="Context limits must be positive."):
        build_invoice_context("doc-1", [], max_units=0)


@pytest.mark.unit
def test_build_invoice_context_enforces_unit_record_and_character_limits() -> None:
    pages = [
        _page_result(unit_id="page:1"),
        _page_result(unit_id="page:2", page_number=2),
    ]

    with pytest.raises(ContextLimitExceeded, match="Too many source units"):
        build_invoice_context("doc-1", pages, max_units=1)

    with pytest.raises(ContextLimitExceeded, match="Too many source records"):
        build_invoice_context(
            "doc-1",
            [
                {
                    "document_id": "doc-1",
                    "unit_id": "sheet:1:row:1",
                    "kind": "sheet_chunk",
                    "status": "completed",
                    "rows": [
                        {"row_number": 1, "cells": [{"value": "A"}]},
                        {"row_number": 2, "cells": [{"value": "B"}]},
                    ],
                }
            ],
            max_records=1,
        )

    with pytest.raises(ContextLimitExceeded, match="context exceeds"):
        build_invoice_context("doc-1", [_page_result()], max_chars=10)
