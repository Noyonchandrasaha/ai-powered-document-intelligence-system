import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.features.document_extraction.data_processing.pipeline import (
    ProcessingPipelineError,
    process_document,
)
from app.features.document_extraction.schemas import DocumentType


def create_page(
    *,
    text: str = "native text",
    needs_ocr: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "page_number": 1,
        "page_type": "digital",
        "extraction_method": "pymupdf",
        "text": text,
        "text_blocks": [],
        "image_coverage": 0.0,
        "needs_ocr": needs_ocr,
        "error": error,
    }


def create_ocr_result(text: str = "ocr text") -> dict[str, Any]:
    return {
        "text": text,
        "confidence": 0.9,
        "words": [],
        "image_width": 100,
        "image_height": 200,
        "extraction_method": "google_vision",
    }


async def collect_results(generator) -> list[dict[str, Any]]:
    return [result async for result in generator]


@pytest.mark.unit
def test_process_document_rejects_invalid_document_id(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "data.csv"
    file_path.write_text("name\nAlice\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Document ID is required"):
        asyncio.run(collect_results(process_document(" ", file_path, DocumentType.CSV)))


@pytest.mark.unit
def test_process_document_rejects_invalid_chunk_size(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "data.csv"
    file_path.write_text("name\nAlice\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Chunk sizes must be positive"):
        asyncio.run(
            collect_results(
                process_document(
                    "document-01",
                    file_path,
                    DocumentType.CSV,
                    csv_chunk_size=0,
                )
            )
        )


@pytest.mark.unit
def test_process_document_rejects_missing_file(workspace_tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Document not found"):
        asyncio.run(
            collect_results(
                process_document(
                    "document-01",
                    workspace_tmp_path / "missing.pdf",
                    DocumentType.PDF,
                )
            )
        )


@pytest.mark.unit
def test_process_pdf_returns_partial_result_without_ocr_client(
    workspace_tmp_path: Path,
):
    file_path = workspace_tmp_path / "document.pdf"
    file_path.write_bytes(b"pdf")

    with (
        patch(
            "app.features.document_extraction.data_processing.pipeline.get_pdf_page_count",
            return_value=1,
        ),
        patch(
            "app.features.document_extraction.data_processing.pipeline.extract_pdf_pages",
            return_value=iter([create_page(needs_ocr=True)]),
        ),
    ):
        results = asyncio.run(
            collect_results(process_document("document-01", file_path, DocumentType.PDF))
        )

    assert results[0]["status"] == "partial"
    assert results[0]["error"] == ("OCR required but Google Vision client is not configured.")
    assert results[0]["metadata"]["ocr_attempted"] is False


@pytest.mark.unit
def test_process_pdf_merges_native_and_ocr_results(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "document.pdf"
    file_path.write_bytes(b"pdf")
    ocr = MagicMock()
    ocr.extract_pdf_page = AsyncMock(return_value=create_ocr_result())

    with (
        patch(
            "app.features.document_extraction.data_processing.pipeline.get_pdf_page_count",
            return_value=1,
        ),
        patch(
            "app.features.document_extraction.data_processing.pipeline.extract_pdf_pages",
            return_value=iter([create_page(needs_ocr=True)]),
        ),
    ):
        results = asyncio.run(
            collect_results(process_document("document-01", file_path, DocumentType.PDF, ocr))
        )

    assert results[0]["status"] == "completed"
    assert results[0]["extraction_method"] == "pymupdf+google_vision"
    assert results[0]["native_text"] == "native text"
    assert results[0]["ocr_text"] == "ocr text"
    assert results[0]["metadata"]["ocr_attempted"] is True
    ocr.extract_pdf_page.assert_awaited_once_with(
        file_path=file_path,
        page_number=1,
    )


@pytest.mark.unit
def test_process_image_requires_ocr_client(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "image.png"
    file_path.write_bytes(b"image")

    with pytest.raises(ProcessingPipelineError, match="required"):
        asyncio.run(collect_results(process_document("document-01", file_path, DocumentType.PNG)))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("file_type", "suffix", "kind", "processor"),
    [
        (DocumentType.XLSX, ".xlsx", "sheet_chunk", "extract_excel_chunks"),
        (DocumentType.CSV, ".csv", "csv_chunk", "extract_csv_chunks"),
    ],
)
def test_process_document_routes_tabular_files(
    workspace_tmp_path: Path,
    file_type: DocumentType,
    suffix: str,
    kind: str,
    processor: str,
):
    file_path = workspace_tmp_path / f"data{suffix}"
    file_path.write_text("data", encoding="utf-8")
    chunk = {
        "sheet_name": "Sheet1",
        "sheet_index": 1,
        "status": "completed",
        "start_row": 1,
        "end_row": 1,
        "rows": [],
        "error": None,
    }
    if file_type is DocumentType.CSV:
        chunk = {
            "encoding": "utf-8",
            "delimiter": ",",
            **chunk,
        }

    with patch(
        f"app.features.document_extraction.data_processing.pipeline.{processor}",
        return_value=iter([chunk]),
    ):
        results = asyncio.run(
            collect_results(process_document("document-01", file_path, file_type))
        )

    assert results[0]["kind"] == kind
    assert results[0]["status"] == "completed"


@pytest.mark.unit
def test_process_document_rejects_unsupported_type(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "data.bin"
    file_path.write_bytes(b"data")

    with pytest.raises(ProcessingPipelineError, match="Unsupported document type"):
        asyncio.run(
            collect_results(
                process_document(
                    "document-01",
                    file_path,
                    cast(DocumentType, "binary"),
                )
            )
        )
