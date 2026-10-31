
"""Source-backed validation for AI invoice extraction."""

import json
import re
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from app.features.invoices.context import InvoiceContext
from app.features.invoices.schemas import (
    ExtractionStatus,
    FieldEvidence,
    FieldStatus,
    InvoiceAIResult,
    InvoiceData,
    InvoiceExtractionResponse,
)

SCALAR_FIELDS = (
    "invoice_number",
    "invoice_date",
    "supplier_name",
    "customer_name",
    "subtotal",
    "tax",
    "total",
    "currency",
)

AMOUNT_FIELDS = {
    "subtotal",
    "tax",
    "total",
    "quantity",
    "unit_price",
    "amount",
}

AMOUNT_PATTERN = re.compile(
    r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)

NUMBER_IN_TEXT = re.compile(
    r"(?<![\w.,])"
    r"(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?![\w.,])"
)

CURRENCY_ALIASES = {
    "BDT": ("BDT", "৳", "TK", "TAKA"),
    "USD": ("USD", "US$", "US DOLLAR"),
    "EUR": ("EUR", "€", "EURO"),
    "GBP": ("GBP", "£", "POUND STERLING"),
    "INR": ("INR", "₹"),
    "CAD": ("CAD", "CA$"),
    "AUD": ("AUD", "A$"),
    "JPY": ("JPY",),
    "CNY": ("CNY", "RMB"),
    "AED": ("AED",),
    "SAR": ("SAR",),
    "SGD": ("SGD",),
    "CHF": ("CHF",),
    "MYR": ("MYR",),
    "PKR": ("PKR",),
    "NZD": ("NZD",),
    "HKD": ("HKD",),
}


class InvoiceValidationError(ValueError):
    """Raised when invoice validation cannot proceed."""


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _load_source_records(
    context: InvoiceContext,
) -> list[dict[str, Any]]:

    records = []

    for line in context.content.splitlines():
        line = line.strip()

        if not line.startswith("{"):
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InvoiceValidationError(
                "Invalid JSONL source record."
            ) from exc

        if (
            not isinstance(record, dict)
            or not record.get("source_id")
        ):
            raise InvoiceValidationError(
                "Invalid source record structure."
            )

        records.append(record)

    if not records:
        raise InvoiceValidationError(
            "No source records available for validation."
        )

    return records


def _verify_source_evidence(
    evidence: FieldEvidence,
    records: list[dict[str, Any]],
) -> str | None:
    """
    Confirm the AI quotation exists in Task 1 data.

    Page evidence must match a page number.
    Spreadsheet evidence must match a cell reference.
    """

    quotation = evidence.source_text

    if not quotation or not quotation.strip():
        return None

    has_page = evidence.page_number is not None
    has_cell = evidence.cell_reference is not None

    if has_page == has_cell:
        return None

    for record in records:

        if has_page:
            if record.get("kind") != "page":
                continue

            if record.get("page_number") != evidence.page_number:
                continue

            original_text = record.get("text")

            if (
                isinstance(original_text, str)
                and _normalize_text(quotation)
                in _normalize_text(original_text)
            ):
                return quotation

        else:
            if record.get("kind") not in {
                "sheet_chunk",
                "csv_chunk",
            }:
                continue

            if record.get("sheet_name") != evidence.sheet_name:
                continue

            for cell in record.get("cells", []):
                if (
                    cell.get("coordinate")
                    != evidence.cell_reference
                ):
                    continue

                original_value = cell.get("value")

                if original_value is None:
                    continue

                if (
                    _normalize_text(quotation)
                    in _normalize_text(str(original_value))
                ):
                    return quotation

    return None


def _verify_numeric_value(
    value: str,
    source_text: str,
) -> bool:

    if not AMOUNT_PATTERN.fullmatch(value):
        return False

    try:
        expected = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return False

    if not expected.is_finite():
        return False

    for match in NUMBER_IN_TEXT.finditer(source_text):

        # A percentage is not automatically a tax amount.
        if source_text[match.end():].lstrip().startswith("%"):
            continue

        try:
            actual = Decimal(
                match.group(1).replace(",", "")
            )
        except InvalidOperation:
            continue

        if actual == expected:
            return True

    return False


def _verify_date(
    value: str,
    source_text: str,
) -> bool:

    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}",
        value,
    ):
        return False

    try:
        expected = date.fromisoformat(value)
    except ValueError:
        return False

    # ISO-formatted dates.
    for match in re.finditer(
        r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)",
        source_text,
    ):
        try:
            if date.fromisoformat(match.group()) == expected:
                return True
        except ValueError:
            pass

    # Numeric dates where day/month order is unambiguous.
    for match in re.finditer(
        r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)",
        source_text,
    ):
        first, second, year = map(int, match.groups())

        if first > 12 and second <= 12:
            day, month = first, second
        elif second > 12 and first <= 12:
            month, day = first, second
        else:
            continue

        try:
            if date(year, month, day) == expected:
                return True
        except ValueError:
            pass

    # Written dates: 30 Oct 2026 or October 30, 2026.
    months = (
        r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
        r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
        r"Aug(?:ust)?|Sep(?:tember)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
    )

    patterns = (
        (
            rf"\b\d{{1,2}}\s+(?:{months})\s+\d{{4}}\b",
            ("%d %b %Y", "%d %B %Y"),
        ),
        (
            rf"\b(?:{months})\s+\d{{1,2}},?\s+\d{{4}}\b",
            (
                "%b %d %Y",
                "%B %d %Y",
                "%b %d, %Y",
                "%B %d, %Y",
            ),
        ),
    )

    for pattern, formats in patterns:
        for match in re.finditer(
            pattern,
            source_text,
            re.IGNORECASE,
        ):
            for date_format in formats:
                try:
                    parsed = datetime.strptime(
                        match.group(),
                        date_format,
                    ).date()

                    if parsed == expected:
                        return True

                except ValueError:
                    continue

    return False


def _verify_currency(
    value: str,
    source_text: str,
) -> bool:

    currency = value.upper()
    aliases = CURRENCY_ALIASES.get(currency)

    if aliases is None:
        return False

    for alias in aliases:

        if not alias.isascii():
            if alias in source_text:
                return True
            continue

        pattern = (
            r"(?<![A-Za-z])"
            + re.escape(alias)
            + r"(?![A-Za-z])"
        )

        if re.search(
            pattern,
            source_text,
            re.IGNORECASE,
        ):
            return True

    return False


def _verify_field_value(
    field_path: str,
    value: str,
    source_text: str,
) -> bool:

    field_name = field_path.rsplit(".", 1)[-1]

    if field_name in AMOUNT_FIELDS:
        return _verify_numeric_value(
            value,
            source_text,
        )

    if field_name == "invoice_date":
        return _verify_date(
            value,
            source_text,
        )

    if field_name == "currency":
        return _verify_currency(
            value,
            source_text,
        )

    return (
        bool(value.strip())
        and _normalize_text(value)
        in _normalize_text(source_text)
    )


def _parse_amount(value: str | None) -> Decimal | None:
    if value is None:
        return None

    try:
        amount = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None

    return amount if amount.is_finite() else None


def _financial_consistency_errors(
    invoice: InvoiceData,
) -> list[str]:
    errors: list[str] = []
    subtotal = _parse_amount(invoice.subtotal)
    tax = _parse_amount(invoice.tax)
    total = _parse_amount(invoice.total)

    if subtotal is not None and tax is not None and total is not None:
        if subtotal + tax != total:
            errors.append(
                "Invoice total does not equal subtotal plus tax."
            )

    line_amounts = [
        _parse_amount(item.amount)
        for item in invoice.line_items
    ]
    confirmed_line_amounts = [
        amount
        for amount in line_amounts
        if amount is not None
    ]

    if (
        subtotal is not None
        and line_amounts
        and len(confirmed_line_amounts) == len(line_amounts)
        and sum(confirmed_line_amounts, Decimal("0")) != subtotal
    ):
        errors.append(
            "Line item amounts do not equal invoice subtotal."
        )

    for index, item in enumerate(invoice.line_items):
        quantity = _parse_amount(item.quantity)
        unit_price = _parse_amount(item.unit_price)
        amount = _parse_amount(item.amount)

        if (
            quantity is not None
            and unit_price is not None
            and amount is not None
            and quantity * unit_price != amount
        ):
            errors.append(
                f"Line item {index} amount does not equal "
                "quantity multiplied by unit price."
            )

    return errors


def _validate_field(
    field_path: str,
    value: str | None,
    evidence_map: dict[str, list[FieldEvidence]],
    records: list[dict[str, Any]],
    incomplete: bool,
) -> tuple[str | None, FieldEvidence]:

    entries = evidence_map.get(field_path, [])

    if value is None:

        is_uncertain = (
            incomplete
            or any(
                entry.status != FieldStatus.MISSING
                for entry in entries
            )
        )

        field_status = (
            FieldStatus.UNCERTAIN
            if is_uncertain
            else FieldStatus.MISSING
        )

        reason = (
            "Value unavailable or source extraction incomplete."
            if is_uncertain
            else "Value not found in available source."
        )

    elif (
        not entries
        or any(
            entry.status != FieldStatus.EXTRACTED
            for entry in entries
        )
    ):
        field_status = FieldStatus.UNCERTAIN
        reason = (
            "Extracted value has missing, conflicting, "
            "or invalid evidence."
        )

    else:
        verified_entries: list[FieldEvidence] = []

        for evidence in entries:
            source_text = _verify_source_evidence(
                evidence,
                records,
            )

            if (
                source_text is not None
                and _verify_field_value(
                    field_path,
                    value,
                    source_text,
                )
            ):
                verified_entries.append(evidence)

        if len(verified_entries) == len(entries):
            field_name = field_path.rsplit(".", 1)[-1]

            if field_name in AMOUNT_FIELDS:
                value = value.replace(",", "")

            elif field_name == "currency":
                value = value.upper()

            return value, verified_entries[0].model_copy(deep=True)

        field_status = FieldStatus.UNCERTAIN

        reason = (
            "Value or source evidence could not be "
            "verified against original document content."
        )

    # Never preserve unsupported model-generated values.
    return None, FieldEvidence(
        field_path=field_path,
        status=field_status,
        source_text=None,
        page_number=None,
        sheet_name=None,
        cell_reference=None,
        reason=reason,
    )


def validate_invoice_extraction(
    document_id: str,
    ai_result: InvoiceAIResult,
    context: InvoiceContext,
) -> InvoiceExtractionResponse:
    """
    Validate AI extraction against original source records.

    Unsupported values become null and are marked uncertain.
    The original AI result is not modified.
    """

    if (
        not document_id
        or document_id != context.document_id
    ):
        raise InvoiceValidationError(
            "Document ID and context ID do not match."
        )

    records = _load_source_records(context)

    if not ai_result.is_invoice:
        return InvoiceExtractionResponse(
            document_id=document_id,
            status=ExtractionStatus.NOT_INVOICE,
            invoice=InvoiceData(
                **{
                    field: None
                    for field in SCALAR_FIELDS
                },
                line_items=[],
            ),
            evidence=[],
            missing_fields=[],
            uncertain_fields=[],
            extracted_at=datetime.now(UTC),
        )

    evidence_map: dict[str, list[FieldEvidence]] = {}

    for evidence in ai_result.evidence:
        evidence_map.setdefault(
            evidence.field_path,
            [],
        ).append(evidence)

    invoice = ai_result.invoice.model_copy(deep=True)

    verified_evidence: list[FieldEvidence] = []
    missing_fields: list[str] = []
    uncertain_fields: list[str] = []

    def validate(
        field_path: str,
        value: str | None,
    ) -> str | None:

        validated, evidence = _validate_field(
            field_path=field_path,
            value=value,
            evidence_map=evidence_map,
            records=records,
            incomplete=context.incomplete,
        )

        verified_evidence.append(evidence)

        if evidence.status == FieldStatus.MISSING:
            missing_fields.append(field_path)

        elif evidence.status == FieldStatus.UNCERTAIN:
            uncertain_fields.append(field_path)

        return validated

    # Validate invoice-level fields.
    for field in SCALAR_FIELDS:
        original_value = getattr(invoice, field)

        setattr(
            invoice,
            field,
            validate(field, original_value),
        )

    # Validate every line item independently.
    verified_items = []

    for index, item in enumerate(invoice.line_items):

        for field in (
            "description",
            "quantity",
            "unit_price",
            "amount",
        ):
            field_path = f"line_items[{index}].{field}"

            setattr(
                item,
                field,
                validate(
                    field_path,
                    getattr(item, field),
                ),
            )

        # Discard line items without any supported fields.
        if any(
            getattr(item, field) is not None
            for field in (
                "description",
                "quantity",
                "unit_price",
                "amount",
            )
        ):
            verified_items.append(item)

    invoice.line_items = verified_items

    validation_errors = _financial_consistency_errors(invoice)

    for error in validation_errors:
        if "total" in error and "total" not in uncertain_fields:
            uncertain_fields.append("total")
        if "subtotal" in error and "subtotal" not in uncertain_fields:
            uncertain_fields.append("subtotal")

    for index, item in enumerate(invoice.line_items):
        if (
            item.amount is not None
            and item.quantity is not None
            and item.unit_price is not None
        ):
            expected = _parse_amount(item.quantity)
            price = _parse_amount(item.unit_price)
            actual = _parse_amount(item.amount)
            if (
                expected is not None
                and price is not None
                and actual is not None
                and expected * price != actual
            ):
                field_path = f"line_items[{index}].amount"
                if field_path not in uncertain_fields:
                    uncertain_fields.append(field_path)

    if not invoice.line_items:
        had_line_items = bool(ai_result.invoice.line_items)
        line_items_status = (
            FieldStatus.UNCERTAIN
            if had_line_items or context.incomplete
            else FieldStatus.MISSING
        )
        line_items_reason = (
            "No line items could be verified against source content."
            if line_items_status == FieldStatus.UNCERTAIN
            else "No line items were found in available source."
        )
        verified_evidence.append(
            FieldEvidence(
                field_path="line_items",
                status=line_items_status,
                source_text=None,
                page_number=None,
                sheet_name=None,
                cell_reference=None,
                reason=line_items_reason,
            )
        )

        if line_items_status == FieldStatus.UNCERTAIN:
            uncertain_fields.append("line_items")
        else:
            missing_fields.append("line_items")

    # Incomplete or uncertain extraction requires review.
    needs_review = bool(
        missing_fields
        or uncertain_fields
        or context.incomplete
    )

    return InvoiceExtractionResponse(
        document_id=document_id,
        status=(
            ExtractionStatus.NEEDS_REVIEW
            if needs_review
            else ExtractionStatus.COMPLETED
        ),
        invoice=invoice,
        evidence=verified_evidence,
        missing_fields=missing_fields,
        uncertain_fields=uncertain_fields,
        validation_errors=validation_errors,
        extracted_at=datetime.now(UTC),
    )
