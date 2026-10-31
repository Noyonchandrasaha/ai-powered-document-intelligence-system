
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FieldStatus(StrEnum):
    EXTRACTED = "extracted"
    MISSING = "missing"
    UNCERTAIN = "uncertain"


class ExtractionStatus(StrEnum):
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    NOT_INVOICE = "not_invoice"
    FAILED = "failed"


class InvoiceLineItem(StrictSchema):
    description: str | None = Field(
        default=None,
        description="Item description, or null if unavailable."
    )
    quantity: str | None = Field(
        default=None,
        description="Quantity as a numeric string, or null."
    )
    unit_price: str | None = Field(
        default=None,
        description="Unit price as a decimal string, or null."
    )
    amount: str | None = Field(
        default=None,
        description="Line total as a decimal string, or null."
    )


class InvoiceData(StrictSchema):
    invoice_number: str | None = Field(
        default=None,
        description="Invoice identifier exactly as shown."
    )
    invoice_date: str | None = Field(
        default=None,
        description=(
            "Unambiguous invoice date in YYYY-MM-DD format. "
            "Use null when missing or ambiguous."
        )
    )
    supplier_name: str | None = Field(
        default=None,
        description="Supplier or seller name."
    )
    customer_name: str | None = Field(
        default=None,
        description="Customer or buyer name."
    )
    subtotal: str | None = Field(
        default=None,
        description=(
            "Subtotal as a decimal string. "
            "Do not calculate missing values."
        )
    )
    tax: str | None = Field(
        default=None,
        description="Total tax amount as a decimal string."
    )
    total: str | None = Field(
        default=None,
        description="Final invoice total as a decimal string."
    )
    currency: str | None = Field(
        default=None,
        description=(
            "Unambiguous ISO 4217 currency code, "
            "such as USD, EUR or BDT."
        )
    )
    line_items: list[InvoiceLineItem] = Field(
        default_factory=list,
        description=(
            "Extracted invoice line items. "
            "Use an empty list when none can be identified."
        )
    )


class FieldEvidence(StrictSchema):
    field_path: str = Field(
        min_length=1,
        description=(
            "Field reference such as invoice_number, "
            "total or line_items[0].amount."
        )
    )
    status: FieldStatus
    source_text: str | None = Field(
        default=None,
        description="Exact supporting text from the document."
    )
    page_number: int | None = Field(
        default=None,
        ge=1,
        description="One-based source page number."
    )
    sheet_name: str | None = Field(
        default=None,
        description="Source spreadsheet sheet name."
    )
    cell_reference: str | None = Field(
        default=None,
        description="Source cell coordinate, such as B12."
    )
    reason: str | None = Field(
        default=None,
        description=(
            "Explanation for missing or uncertain information."
        )
    )


class InvoiceAIResult(StrictSchema):
    """
    Structured output returned by the AI model.
    """

    is_invoice: bool = Field(
        description="Whether the supplied content is an invoice."
    )
    invoice: InvoiceData
    evidence: list[FieldEvidence] = Field(
        description="Source evidence and field-level uncertainty."
    )


class InvoiceExtractionResponse(StrictSchema):
    """
    Final API response after application-level validation.
    """

    document_id: str = Field(min_length=1)
    status: ExtractionStatus
    invoice: InvoiceData
    evidence: list[FieldEvidence]
    missing_fields: list[str] = Field(default_factory=list)
    uncertain_fields: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    extracted_at: datetime
