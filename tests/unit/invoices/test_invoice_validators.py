import json
from dataclasses import replace

import pytest
from app.features.invoices.context import InvoiceContext
from app.features.invoices.schemas import (
    ExtractionStatus,
    FieldEvidence,
    FieldStatus,
    InvoiceAIResult,
    InvoiceData,
    InvoiceLineItem,
)
from app.features.invoices.validators import (
    InvoiceValidationError,
    _financial_consistency_errors,
    validate_invoice_extraction,
)


def _context(
    text: str = (
        "Invoice INV-100 Date 31/10/2026 Supplier Acme "
        "Customer Beta Subtotal 1,000.00 Tax 150.00 "
        "Total 1,150.00 Currency EUR € Widget 2 500.00 1,000.00"
    ),
    *,
    incomplete: bool = False,
) -> InvoiceContext:
    return InvoiceContext(
        document_id="doc-1",
        content=json.dumps(
            {
                "source_id": "page:1:native",
                "kind": "page",
                "page_number": 1,
                "text": text,
            },
            ensure_ascii=False,
        )
        + "\n",
        source_units=("page:1",),
        source_record_count=1,
        incomplete=incomplete,
        warnings=(),
    )


def _evidence(field_path: str, source_text: str) -> FieldEvidence:
    return FieldEvidence(
        field_path=field_path,
        status=FieldStatus.EXTRACTED,
        source_text=source_text,
        page_number=1,
    )


@pytest.mark.unit
def test_validate_invoice_extraction_accepts_source_backed_values() -> None:
    result = validate_invoice_extraction(
        "doc-1",
        InvoiceAIResult(
            is_invoice=True,
            invoice=InvoiceData(
                invoice_number="INV-100",
                invoice_date="2026-10-31",
                supplier_name="Acme",
                customer_name="Beta",
                subtotal="1,000.00",
                tax="150.00",
                total="1,150.00",
                currency="EUR",
                line_items=[
                    InvoiceLineItem(
                        description="Widget",
                        quantity="2",
                        unit_price="500.00",
                        amount="1,000.00",
                    )
                ],
            ),
            evidence=[
                _evidence("invoice_number", "INV-100"),
                _evidence("invoice_date", "31/10/2026"),
                _evidence("supplier_name", "Acme"),
                _evidence("customer_name", "Beta"),
                _evidence("subtotal", "1,000.00"),
                _evidence("tax", "150.00"),
                _evidence("total", "1,150.00"),
                _evidence("currency", "€"),
                _evidence("line_items[0].description", "Widget"),
                _evidence("line_items[0].quantity", "2"),
                _evidence("line_items[0].unit_price", "500.00"),
                _evidence("line_items[0].amount", "1,000.00"),
            ],
        ),
        _context(),
    )

    assert result.status is ExtractionStatus.COMPLETED
    assert result.invoice.subtotal == "1000.00"
    assert result.invoice.total == "1150.00"
    assert result.invoice.currency == "EUR"
    assert result.invoice.line_items[0].amount == "1000.00"
    assert result.missing_fields == []
    assert result.uncertain_fields == []


@pytest.mark.unit
def test_financial_consistency_detects_total_and_line_item_mismatches() -> None:
    errors = _financial_consistency_errors(
        InvoiceData(
            subtotal="100.00",
            tax="10.00",
            total="120.00",
            line_items=[
                InvoiceLineItem(
                    quantity="2",
                    unit_price="40.00",
                    amount="90.00",
                )
            ],
        )
    )

    assert errors == [
        "Invoice total does not equal subtotal plus tax.",
        "Line item amounts do not equal invoice subtotal.",
        "Line item 0 amount does not equal quantity multiplied by unit price.",
    ]


@pytest.mark.unit
def test_validate_invoice_extraction_cross_verifies_pdf_and_spreadsheet() -> None:
    context = InvoiceContext(
        document_id="doc-1",
        content=(
            json.dumps(
                {
                    "source_id": "page:1:native",
                    "kind": "page",
                    "page_number": 1,
                    "text": "Total 100.00",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "source_id": "sheet:1:row:2",
                    "kind": "sheet_chunk",
                    "sheet_name": "Sheet1",
                    "cells": [{"coordinate": "B2", "value": "100.00"}],
                }
            )
            + "\n"
        ),
        source_units=("page:1", "sheet:1:row:2"),
        source_record_count=2,
        incomplete=False,
        warnings=(),
    )
    result = validate_invoice_extraction(
        "doc-1",
        InvoiceAIResult(
            is_invoice=True,
            invoice=InvoiceData(total="100.00"),
            evidence=[
                _evidence("total", "100.00"),
                FieldEvidence(
                    field_path="total",
                    status=FieldStatus.EXTRACTED,
                    source_text="100.00",
                    sheet_name="Sheet1",
                    cell_reference="B2",
                ),
            ],
        ),
        context,
    )

    assert result.invoice.total == "100.00"
    assert "total" not in result.uncertain_fields


@pytest.mark.unit
def test_validate_invoice_extraction_removes_unsupported_values() -> None:
    ai_result = InvoiceAIResult(
        is_invoice=True,
        invoice=InvoiceData(total="999.00"),
        evidence=[_evidence("total", "999.00")],
    )

    result = validate_invoice_extraction("doc-1", ai_result, _context())

    assert ai_result.invoice.total == "999.00"
    assert result.invoice.total is None
    assert result.status is ExtractionStatus.NEEDS_REVIEW
    assert "total" in result.uncertain_fields


@pytest.mark.unit
def test_validate_invoice_extraction_adds_line_items_evidence_when_missing() -> None:
    result = validate_invoice_extraction(
        "doc-1",
        InvoiceAIResult(
            is_invoice=True,
            invoice=InvoiceData(),
            evidence=[],
        ),
        _context(),
    )

    line_items_evidence = [entry for entry in result.evidence if entry.field_path == "line_items"]

    assert len(line_items_evidence) == 1
    assert line_items_evidence[0].status is FieldStatus.MISSING
    assert "line_items" in result.missing_fields


@pytest.mark.unit
def test_validate_invoice_extraction_marks_invalid_line_items_uncertain() -> None:
    result = validate_invoice_extraction(
        "doc-1",
        InvoiceAIResult(
            is_invoice=True,
            invoice=InvoiceData(line_items=[InvoiceLineItem(description="Unknown item")]),
            evidence=[_evidence("line_items[0].description", "Unknown item")],
        ),
        _context(),
    )

    assert result.invoice.line_items == []
    assert "line_items[0].description" in result.uncertain_fields
    assert "line_items" in result.uncertain_fields


@pytest.mark.unit
def test_validate_invoice_extraction_returns_not_invoice_status() -> None:
    result = validate_invoice_extraction(
        "doc-1",
        InvoiceAIResult(
            is_invoice=False,
            invoice=InvoiceData(total="not used"),
            evidence=[],
        ),
        _context(),
    )

    assert result.status is ExtractionStatus.NOT_INVOICE
    assert result.invoice.total is None
    assert result.invoice.line_items == []
    assert result.evidence == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("document_id", "context", "message"),
    [
        ("other", _context(), "Document ID and context ID do not match."),
        (
            "doc-1",
            replace(_context(), content="not json"),
            "No source records available for validation.",
        ),
    ],
)
def test_validate_invoice_extraction_rejects_invalid_context(
    document_id: str,
    context: InvoiceContext,
    message: str,
) -> None:
    with pytest.raises(InvoiceValidationError, match=message):
        validate_invoice_extraction(
            document_id,
            InvoiceAIResult(
                is_invoice=False,
                invoice=InvoiceData(),
                evidence=[],
            ),
            context,
        )
