import asyncio
from functools import wraps
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from app.features.invoices.context import InvoiceContext
from app.features.invoices.extractor import (
    InvoiceExtractionRefusalError,
    InvoiceExtractionResponseError,
    OpenAIInvoiceExtractor,
)
from app.features.invoices.schemas import InvoiceAIResult, InvoiceData


def _run_async(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


def _context() -> InvoiceContext:
    return InvoiceContext(
        document_id="doc-1",
        content='{"source_id":"page:1:native","text":"Invoice"}\n',
        source_units=("page:1",),
        source_record_count=1,
        incomplete=False,
        warnings=(),
    )


def _result() -> InvoiceAIResult:
    return InvoiceAIResult(
        is_invoice=False,
        invoice=InvoiceData(),
        evidence=[],
    )


class FakeClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.responses = SimpleNamespace(
            parse=AsyncMock(return_value=response)
        )
        self.embeddings = SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])]
                )
            )
        )

    def with_options(self, **_: object) -> "FakeClient":
        return self


def _response(
    *,
    parsed: InvoiceAIResult | None = None,
    status: str = "completed",
    output: list[object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        id="response-1",
        output=output or [],
        output_parsed=parsed,
    )


@pytest.mark.unit
def test_extractor_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="model is required"):
        OpenAIInvoiceExtractor(FakeClient(_response()), model=" ")

    with pytest.raises(ValueError, match="at least 256"):
        OpenAIInvoiceExtractor(FakeClient(_response()), max_output_tokens=128)

    with pytest.raises(ValueError, match="timeout_seconds"):
        OpenAIInvoiceExtractor(FakeClient(_response()), timeout_seconds=0)

    with pytest.raises(ValueError, match="max_retries"):
        OpenAIInvoiceExtractor(FakeClient(_response()), max_retries=-1)


@pytest.mark.unit
@_run_async
async def test_extractor_returns_structured_result() -> None:
    client = FakeClient(_response(parsed=_result()))
    extractor = OpenAIInvoiceExtractor(client, model="gpt-test")

    result = await extractor.extract(_context())

    assert result == _result()
    client.responses.parse.assert_awaited_once()


@pytest.mark.unit
@_run_async
async def test_extractor_rejects_incomplete_response() -> None:
    client = FakeClient(_response(parsed=_result(), status="incomplete"))
    extractor = OpenAIInvoiceExtractor(client)

    with pytest.raises(InvoiceExtractionResponseError, match="did not complete"):
        await extractor.extract(_context())


@pytest.mark.unit
@_run_async
async def test_extractor_rejects_empty_parsed_response() -> None:
    client = FakeClient(_response(parsed=None))
    extractor = OpenAIInvoiceExtractor(client)

    with pytest.raises(InvoiceExtractionResponseError, match="no parsed"):
        await extractor.extract(_context())


@pytest.mark.unit
@_run_async
async def test_extractor_handles_model_refusal() -> None:
    refusal = SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="refusal")],
    )
    client = FakeClient(
        _response(parsed=_result(), output=[refusal])
    )
    extractor = OpenAIInvoiceExtractor(client)

    with pytest.raises(InvoiceExtractionRefusalError, match="refused"):
        await extractor.extract(_context())


@pytest.mark.unit
@_run_async
async def test_extractor_creates_embedding() -> None:
    client = FakeClient(_response())
    extractor = OpenAIInvoiceExtractor(
        client,
        model="gpt-test",
        embedding_model="embedding-test",
    )

    vector = await extractor.embed("invoice total 100")

    assert vector == [0.1, 0.2, 0.3]
    client.embeddings.create.assert_awaited_once_with(
        model="embedding-test",
        input="invoice total 100",
        encoding_format="float",
    )
