from datetime import UTC, datetime

import pytest
from app.features.invoices.schemas import (
    ExtractionStatus,
    FieldEvidence,
    FieldStatus,
    InvoiceAIResult,
    InvoiceData,
    InvoiceExtractionResponse,
)
from pydantic import ValidationError


@pytest.mark.unit
def test_invoice_data_uses_safe_defaults_for_missing_values() -> None:
    data = InvoiceData()

    assert data.invoice_number is None
    assert data.invoice_date is None
    assert data.line_items == []


@pytest.mark.unit
def test_invoice_ai_result_accepts_minimal_non_invoice_result() -> None:
    result = InvoiceAIResult(
        is_invoice=False,
        invoice=InvoiceData(),
        evidence=[],
    )

    assert result.is_invoice is False
    assert result.evidence == []


@pytest.mark.unit
def test_invoice_extraction_response_uses_empty_field_lists_by_default() -> None:
    result = InvoiceExtractionResponse(
        document_id="doc-1",
        status=ExtractionStatus.COMPLETED,
        invoice=InvoiceData(),
        evidence=[],
        extracted_at=datetime.now(UTC),
    )

    assert result.missing_fields == []
    assert result.uncertain_fields == []


@pytest.mark.unit
def test_invoice_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        InvoiceData(unexpected="value")


@pytest.mark.unit
def test_field_evidence_rejects_non_positive_page_number() -> None:
    with pytest.raises(ValidationError):
        FieldEvidence(
            field_path="total",
            status=FieldStatus.EXTRACTED,
            page_number=0,
        )
