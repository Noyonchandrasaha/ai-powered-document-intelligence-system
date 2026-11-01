"""Conservative invoice grouping; ambiguous groups are exposed for human review."""

import json
import re
from collections.abc import Iterable, Iterator
from typing import Any

from app.features.invoices.context import MAX_CONTEXT_CHARS, MAX_SOURCE_UNITS

INVOICE_NUMBER = re.compile(
    r"\binvoice\s*(?:number|no\.?|#)\s*[:#-]?\s*([A-Za-z0-9][A-Za-z0-9_/-]*)",
    re.IGNORECASE,
)


class InvoiceGrouper:
    """Bounded state shared across cursor batches and workbook chunks."""

    def __init__(self) -> None:
        self.current: list[dict[str, Any]] = []
        self.key: str | None = None
        self.size = 0
        self.index = 0
        self.headers: dict[str, list[dict[str, Any]]] = {}

    def flush(self, *, uncertain: bool = False) -> dict[str, Any] | None:
        if not self.current:
            return None
        self.index += 1
        result = {
            "candidate_id": f"candidate:{self.index}",
            "invoice_hint": self.key,
            "unit_ids": [unit["unit_id"] for unit in self.current],
            "requires_review": uncertain or self.key is None,
            "units": self.current,
        }
        self.current, self.size, self.key = [], 0, None
        return result

    def feed(self, unit: dict[str, Any]) -> Iterator[dict[str, Any]]:
        tabular = unit.get("kind") in {"sheet_chunk", "csv_chunk"}
        sheet_key = str(unit.get("sheet_index") or unit.get("kind"))
        if tabular:
            rows = unit.get("rows", [])
            if rows and sheet_key not in self.headers:
                header_values = [
                    str(cell.get("value", "")).strip() for cell in rows[0].get("cells", [])
                ]
                header_fields = sum(
                    bool(
                        re.fullmatch(
                            r"(?:invoice[ _]*(?:number|no\.?|#|date)|"
                            r"supplier[ _]*name|total|currency)",
                            value,
                            re.I,
                        )
                    )
                    for value in header_values
                )
                if header_fields >= 2 and any(
                    re.fullmatch(
                        r"invoice[ _]*(?:number|no\.?|#)", str(cell.get("value", "")).strip(), re.I
                    )
                    for cell in rows[0].get("cells", [])
                ):
                    self.headers[sheet_key] = [rows[0]]
            header = self.headers.get(sheet_key)
            if header:
                previous = self.flush()
                if previous:
                    yield previous
                for row in rows:
                    if row == header[0]:
                        continue
                    self.index += 1
                    yield {
                        "candidate_id": f"candidate:{self.index}",
                        "invoice_hint": None,
                        "unit_ids": [unit["unit_id"]],
                        "requires_review": True,
                        "units": [{**unit, "rows": [*header, row]}],
                    }
                return
        text = unit.get("text") or unit.get("native_text") or unit.get("ocr_text") or ""
        matches = list(dict.fromkeys(INVOICE_NUMBER.findall(text)))
        key = matches[0].casefold() if len(matches) == 1 else None
        if tabular:
            key = f"sheet:{sheet_key}"
        unit_size = len(json.dumps(unit, default=str))
        limit = (
            len(self.current) >= MAX_SOURCE_UNITS or self.size + unit_size > MAX_CONTEXT_CHARS // 2
        )
        if self.current and ((key is not None and key != self.key) or limit):
            previous = self.flush(uncertain=limit)
            if previous:
                yield previous
        if not self.current:
            self.key = key
        self.current.append(unit)
        self.size += unit_size
        if len(matches) > 1:
            self.key = None


def group_invoice_units(units: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    grouper = InvoiceGrouper()
    for unit in units:
        yield from grouper.feed(unit)
    last = grouper.flush()
    if last:
        yield last
