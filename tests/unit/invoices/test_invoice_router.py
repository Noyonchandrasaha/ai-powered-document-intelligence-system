import asyncio
from functools import wraps
from unittest.mock import AsyncMock, MagicMock

import pytest
from app.features.invoices.router import (
    ExtractInvoiceRequest,
    get_invoice_extraction,
    get_latest_invoice_extraction,
    load_document_results,
)
from fastapi import HTTPException
from pydantic import ValidationError


def _run_async(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


class FakeCursor:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self.results = results
        self.limit_value: int | None = None

    def limit(self, value: int) -> "FakeCursor":
        self.limit_value = value
        return self

    async def to_list(self, length: int) -> list[dict[str, object]]:
        assert self.limit_value == length
        return self.results


def _database(results: list[dict[str, object]]) -> MagicMock:
    collection = MagicMock()
    collection.find.return_value = FakeCursor(results)
    database = MagicMock()
    database.__getitem__.return_value = collection
    return database


@pytest.mark.unit
def test_extract_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ExtractInvoiceRequest(unknown="value")


@pytest.mark.unit
def test_extract_request_accepts_unit_and_extraction_ids() -> None:
    payload = ExtractInvoiceRequest(
        unit_ids=["page:1"],
        extraction_id="a" * 32,
    )

    assert payload.unit_ids == ["page:1"]
    assert payload.extraction_id == "a" * 32


@pytest.mark.unit
@_run_async
async def test_load_document_results_filters_and_returns_results() -> None:
    results = [
        {"document_id": "doc-1", "unit_id": "page:1"},
    ]
    database = _database(results)

    loaded = await load_document_results(
        database,
        "doc-1",
        ["page:1"],
    )

    assert loaded == results
    collection = database.__getitem__.return_value
    collection.find.assert_called_once_with(
        {
            "document_id": "doc-1",
            "unit_id": {"$in": ["page:1"]},
        },
        {"_id": 0},
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("unit_ids", "results", "status_code", "message"),
    [
        (
            ["page:1", "page:1"],
            [],
            422,
            "Duplicate source unit IDs.",
        ),
        (
            ["page:1", "page:2"],
            [{"unit_id": "page:1"}],
            422,
            "One or more requested source units were not found.",
        ),
        (
            None,
            [],
            404,
            "No Task 1 extraction results found for this document.",
        ),
    ],
)
@_run_async
async def test_load_document_results_rejects_invalid_selection(
    unit_ids: list[str] | None,
    results: list[dict[str, object]],
    status_code: int,
    message: str,
) -> None:
    with pytest.raises(HTTPException) as error:
        await load_document_results(
            _database(results),
            "doc-1",
            unit_ids,
        )

    assert error.value.status_code == status_code
    assert error.value.detail == message


@pytest.mark.unit
@_run_async
async def test_get_invoice_extraction_returns_not_found_when_missing() -> None:
    repository = MagicMock()
    repository.get_extraction = AsyncMock(return_value=None)

    with pytest.raises(HTTPException) as error:
        await get_invoice_extraction("a" * 32, repository)

    assert error.value.status_code == 404
    assert error.value.detail == "Invoice extraction not found."


@pytest.mark.unit
@_run_async
async def test_get_latest_invoice_extraction_returns_not_found_when_missing() -> None:
    repository = MagicMock()
    repository.get_latest_extraction = AsyncMock(return_value=None)

    with pytest.raises(HTTPException) as error:
        await get_latest_invoice_extraction("a" * 32, repository)

    assert error.value.status_code == 404
    assert error.value.detail == "No invoice extraction found."
