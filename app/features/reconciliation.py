"""Deterministic PDF/PDF and PDF/spreadsheet invoice reconciliation."""

from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any
from uuid import uuid4

from app.core.database import get_database
from app.features.document_extraction.repository import DocumentRepository
from app.features.invoices.schemas import InvoiceExtractionResponse
from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field
from pymongo.asynchronous.database import AsyncDatabase

router = APIRouter(prefix="/reconciliations", tags=["Cross verification"])
Database = Annotated[AsyncDatabase[Any], Depends(get_database)]


class ReconcileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left_document_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    right_document_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    amount_tolerance: Decimal = Field(default=Decimal("0.01"), ge=0, le=1)


def normalize(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def decimal(value: str | None) -> Decimal | None:
    try:
        amount = Decimal((value or "").replace(",", ""))
        return amount if amount.is_finite() else None
    except InvalidOperation:
        return None


def compare_invoices(
    left: InvoiceExtractionResponse,
    right: InvoiceExtractionResponse,
    tolerance: Decimal = Decimal("0.01"),
) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    for field in (
        "invoice_number",
        "invoice_date",
        "supplier_name",
        "customer_name",
        "currency",
        "subtotal",
        "tax",
        "total",
    ):
        a, b = getattr(left.invoice, field), getattr(right.invoice, field)
        if a is None or b is None:
            reason = "missing_value"
        elif field in {"subtotal", "tax", "total"}:
            first, second = decimal(a), decimal(b)
            if first is not None and second is not None and abs(first - second) <= tolerance:
                continue
            reason = "amount_mismatch"
        elif normalize(a) == normalize(b):
            continue
        else:
            reason = "value_mismatch"
        differences.append({"field": field, "left": a, "right": b, "reason": reason})

    def items(result: InvoiceExtractionResponse) -> Counter[tuple[str, ...]]:
        return Counter(
            tuple(
                normalize(item.description)
                if field == "description"
                else str(decimal(getattr(item, field)))
                for field in ("description", "quantity", "unit_price", "amount")
            )
            for item in result.invoice.line_items
        )

    if not left.invoice.line_items or not right.invoice.line_items or items(left) != items(right):
        differences.append({"field": "line_items", "reason": "missing_or_different_line_items"})
    review = left.status.value != "completed" or right.status.value != "completed"
    return {
        "left_extraction_id": left.extraction_id,
        "right_extraction_id": right.extraction_id,
        "invoice_number": left.invoice.invoice_number,
        "status": "mismatch" if differences else "needs_review" if review else "matched",
        "differences": differences,
    }


def reconcile(
    left: list[InvoiceExtractionResponse],
    right: list[InvoiceExtractionResponse],
    tolerance: Decimal,
) -> list[dict[str, Any]]:
    sides: list[dict[str, list[InvoiceExtractionResponse]]] = []
    for records in (left, right):
        by_number: dict[str, list[InvoiceExtractionResponse]] = defaultdict(list)
        for record in records:
            by_number[normalize(record.invoice.invoice_number)].append(record)
        sides.append(by_number)
    rows: list[dict[str, Any]] = []
    for number in sorted(sides[0].keys() | sides[1].keys()):
        a, b = sides[0].get(number, []), sides[1].get(number, [])
        if not number or len(a) > 1 or len(b) > 1:
            rows.append(
                {
                    "invoice_number": number or None,
                    "status": "needs_review",
                    "reason": "missing_or_ambiguous_invoice_number",
                    "left_extraction_ids": [item.extraction_id for item in a],
                    "right_extraction_ids": [item.extraction_id for item in b],
                }
            )
        elif not a or not b:
            rows.append(
                {
                    "invoice_number": number,
                    "status": "unmatched",
                    "missing_from": "left" if not a else "right",
                }
            )
        else:
            rows.append(compare_invoices(a[0], b[0], tolerance))
    return rows


@router.post("", status_code=201)
async def reconcile_documents(payload: ReconcileRequest, database: Database) -> dict[str, Any]:
    ids = [payload.left_document_id, payload.right_document_id]
    if ids[0] == ids[1]:
        raise HTTPException(422, "Choose two different documents.")
    for document_id in ids:
        if await DocumentRepository(database).get_document(document_id) is None:
            raise HTTPException(404, f"Document not found: {document_id}")
    sides: list[list[InvoiceExtractionResponse]] = []
    for document_id in ids:
        # A re-run replaces the same source group logically; preserve separate
        # groups even if they share an invoice number, so ambiguity is visible.
        cursor = await database["invoice_extractions"].aggregate(
            [
                {"$match": {"document_id": document_id}},
                {"$sort": {"extracted_at": -1}},
                {
                    "$group": {
                        "_id": {"$ifNull": ["$source_fingerprint", "$source_units"]},
                        "latest": {"$first": "$$ROOT"},
                    }
                },
                {"$replaceRoot": {"newRoot": "$latest"}},
                {"$project": {field: 1 for field in InvoiceExtractionResponse.model_fields}},
                {"$limit": 10_001},
            ]
        )
        records = await cursor.to_list(length=10_001)
        if not records:
            raise HTTPException(409, f"Extract invoices first: {document_id}")
        if len(records) > 10_000:
            raise HTTPException(422, "Reconcile smaller document groups (maximum 10,000 invoices).")
        for record in records:
            record.pop("_id", None)
        sides.append([InvoiceExtractionResponse.model_validate(record) for record in records])
    rows = reconcile(sides[0], sides[1], payload.amount_tolerance)
    counts = dict(Counter(row["status"] for row in rows))
    report = {
        "reconciliation_id": uuid4().hex,
        "document_ids": ids,
        "amount_tolerance": str(payload.amount_tolerance),
        "created_at": datetime.now(UTC),
        "status": "matched"
        if rows and all(r["status"] == "matched" for r in rows)
        else "needs_review",
        "counts": counts,
        "rows": rows,
    }
    await database["reconciliations"].insert_one(dict(report))
    return report


@router.get("/{reconciliation_id}")
async def get_reconciliation(
    reconciliation_id: Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")],
    database: Database,
) -> dict[str, Any]:
    report = await database["reconciliations"].find_one(
        {"reconciliation_id": reconciliation_id},
        {"_id": 0},
    )
    if report is None:
        raise HTTPException(404, "Reconciliation not found.")
    return dict(report)
