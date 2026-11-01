import asyncio
from datetime import UTC, datetime
from functools import wraps
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from app.features.invoices.repository import (
    RESULT_PROJECTION,
    InvoiceRepository,
    InvoiceRepositoryError,
)
from app.features.invoices.schemas import (
    ExtractionStatus,
    InvoiceData,
    InvoiceExtractionResponse,
)


def _result() -> InvoiceExtractionResponse:
    return InvoiceExtractionResponse(
        document_id="doc-1",
        status=ExtractionStatus.COMPLETED,
        invoice=InvoiceData(total="100.00"),
        evidence=[],
        extracted_at=datetime(2026, 10, 31, tzinfo=UTC),
    )


def _repository() -> tuple[InvoiceRepository, AsyncMock]:
    collection = AsyncMock()
    database = MagicMock()
    database.__getitem__.return_value = collection
    return InvoiceRepository(database), collection


class _Cursor:
    def __init__(self, documents: list[dict[str, object]]) -> None:
        self.documents = documents

    async def to_list(self, length: int) -> list[dict[str, object]]:
        return self.documents[:length]

    def __aiter__(self):
        async def iterate():
            for document in self.documents:
                yield document

        return iterate()


def _semantic_repository() -> InvoiceRepository:
    stored = _result().model_dump(mode="json")
    stored["embedding"] = [1.0, 0.0]
    collection = MagicMock()
    collection.find.return_value = _Cursor([stored])
    database = MagicMock()
    database.__getitem__.return_value = collection
    return InvoiceRepository(database)


def _run_async(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


@pytest.mark.unit
@_run_async
async def test_ensure_indexes_creates_unique_and_lookup_indexes() -> None:
    repository, collection = _repository()

    await repository.ensure_indexes()

    assert collection.create_index.await_args_list == [
        call(
            [("extraction_id", 1)],
            unique=True,
            name="uq_extraction_id",
        ),
        call(
            [
                ("document_id", 1),
                ("extracted_at", -1),
                ("extraction_id", -1),
            ],
            name="idx_document_extractions",
        ),
        call("identity", unique=True),
    ]


@pytest.mark.unit
@_run_async
async def test_save_extraction_persists_json_compatible_payload() -> None:
    repository, collection = _repository()

    extraction_id = await repository.save_extraction(
        extraction_id="ext-1",
        result=_result(),
        model_name="gpt-test",
        prompt_version="1.0.0",
        source_units=("page:1",),
    )

    assert extraction_id == "ext-1"
    collection.update_one.assert_awaited_once()

    query, update = collection.update_one.await_args.args[:2]
    assert query == {
        "extraction_id": "ext-1",
        "document_id": "doc-1",
    }
    payload = update["$setOnInsert"]
    assert payload["invoice"]["total"] == "100.00"
    assert payload["extraction_id"] == "ext-1"
    assert payload["source_units"] == ["page:1"]
    assert payload["extracted_at"] == datetime(2026, 10, 31, tzinfo=UTC)
    assert payload["created_at"].tzinfo is UTC
    assert collection.update_one.await_args.kwargs == {"upsert": True}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"extraction_id": " "}, "Extraction ID is required."),
        ({"model_name": " "}, "AI model name is required."),
        ({"prompt_version": " "}, "Prompt version is required."),
    ],
)
@_run_async
async def test_save_extraction_rejects_missing_metadata(
    kwargs: dict[str, str],
    message: str,
) -> None:
    repository, collection = _repository()
    values = {
        "extraction_id": "ext-1",
        "result": _result(),
        "model_name": "gpt-test",
        "prompt_version": "1.0.0",
        "source_units": ("page:1",),
    }
    values.update(kwargs)

    with pytest.raises(InvoiceRepositoryError, match=message):
        await repository.save_extraction(**values)

    collection.update_one.assert_not_awaited()


@pytest.mark.unit
@_run_async
async def test_save_extraction_rejects_oversized_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    repository, collection = _repository()
    monkeypatch.setattr(
        "app.features.invoices.repository.MAX_DOCUMENT_BYTES",
        0,
    )

    with pytest.raises(
        InvoiceRepositoryError,
        match="exceeds storage size limit",
    ):
        await repository.save_extraction(
            extraction_id="ext-1",
            result=_result(),
            model_name="gpt-test",
            prompt_version="1.0.0",
            source_units=(),
        )

    collection.update_one.assert_not_awaited()


@pytest.mark.unit
@_run_async
async def test_get_extraction_returns_validated_result() -> None:
    repository, collection = _repository()
    stored = _result().model_dump(mode="json")
    collection.find_one.return_value = stored

    result = await repository.get_extraction("ext-1")

    assert result == _result()
    collection.find_one.assert_awaited_once_with(
        {"extraction_id": "ext-1"},
        projection=RESULT_PROJECTION,
    )


@pytest.mark.unit
@_run_async
async def test_get_latest_and_metadata_return_none_or_raw_document() -> None:
    repository, collection = _repository()
    collection.find_one.side_effect = [None, {"document_id": "doc-1"}]

    latest = await repository.get_latest_extraction("doc-1")
    metadata = await repository.get_extraction_metadata("ext-1")

    assert latest is None
    assert metadata == {"document_id": "doc-1"}


@pytest.mark.unit
@_run_async
async def test_semantic_search_ranks_stored_embeddings() -> None:
    repository = _semantic_repository()

    results = await repository.semantic_search([1.0, 0.0], limit=1)

    assert len(results) == 1
    assert results[0].document_id == "doc-1"


@pytest.mark.unit
@_run_async
async def test_semantic_search_rejects_zero_query_vector() -> None:
    repository = _semantic_repository()

    with pytest.raises(InvoiceRepositoryError, match="cannot be zero"):
        await repository.semantic_search([0.0, 0.0])
