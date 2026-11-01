"""Evaluate invoice extraction results against a labeled JSONL dataset.

Each line must contain:

{"expected": {"invoice_number": "INV-1", ...},
 "actual": {"invoice_number": "INV-1", ...},
 "status": "completed"}

The evaluator intentionally uses exact normalized matches. It does not hide
missing labels or convert uncertain values into correct predictions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

FIELD_NAMES = (
    "invoice_number",
    "invoice_date",
    "supplier_name",
    "customer_name",
    "subtotal",
    "tax",
    "total",
    "currency",
)


def _normalize(value: Any, *, numeric: bool = False) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().casefold().split()) or None
    if numeric and text:
        try:
            amount = Decimal(text.replace(",", ""))
            if amount.is_finite():
                return str(amount.normalize())
        except InvalidOperation:
            pass
    return text


def _line_item_key(item: Any) -> tuple[str | None, ...]:
    if not isinstance(item, dict):
        return (None,)
    return tuple(
        _normalize(item.get(field), numeric=field != "description")
        for field in ("description", "quantity", "unit_price", "amount")
    )


def _line_item_scores(expected: Any, actual: Any) -> tuple[int, int, int]:
    expected_list = expected if isinstance(expected, list) else []
    actual_list = actual if isinstance(actual, list) else []
    expected_items = Counter(_line_item_key(item) for item in expected_list)
    actual_items = Counter(_line_item_key(item) for item in actual_list)
    true_positive = sum((expected_items & actual_items).values())
    return true_positive, sum(actual_items.values()), sum(expected_items.values())


def evaluate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    field_metrics: dict[str, dict[str, int | float]] = {}

    for field in FIELD_NAMES:
        labeled = 0
        correct = 0
        for row in rows:
            expected = row.get("expected", {})
            actual = row.get("actual", {})
            if field not in expected:
                continue
            numeric = field in {"subtotal", "tax", "total"}
            expected_value = _normalize(expected.get(field), numeric=numeric)
            labeled += 1
            if expected_value == _normalize(actual.get(field), numeric=numeric):
                correct += 1
        field_metrics[field] = {
            "labeled": labeled,
            "correct": correct,
            "accuracy": correct / labeled if labeled else 0.0,
        }

    true_positive = predicted = expected_count = 0
    for row in rows:
        expected = row.get("expected", {})
        actual = row.get("actual", {})
        current_true_positive, current_predicted, current_expected = _line_item_scores(
            expected.get("line_items", []),
            actual.get("line_items", []),
        )
        true_positive += current_true_positive
        predicted += current_predicted
        expected_count += current_expected

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / expected_count if expected_count else 0.0
    total = len(rows)
    times = sorted(
        float(row["duration_seconds"])
        for row in rows
        if isinstance(row.get("duration_seconds"), (int, float))
    )
    return {
        "documents": total,
        "field_metrics": field_metrics,
        "line_item_metrics": {
            "true_positive": true_positive,
            "predicted": predicted,
            "expected": expected_count,
            "precision": precision,
            "recall": recall,
        },
        "status_counts": status_counts,
        "success_rate": status_counts.get("completed", 0) / total if total else 0.0,
        "failure_rate": status_counts.get("failed", 0) / total if total else 0.0,
        "partial_rate": status_counts.get("partial", 0) / total if total else 0.0,
        "latency_seconds": {
            "samples": len(times),
            "p50": times[math.ceil(len(times) * 0.50) - 1] if times else None,
            "p95": times[math.ceil(len(times) * 0.95) - 1] if times else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Labeled JSONL file")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        args.dataset.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Line {line_number} must contain a JSON object.")
        rows.append(value)

    print(json.dumps(evaluate(rows), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
