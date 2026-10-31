import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pymupdf
import pytest
from app.features.document_extraction.data_processing.pipeline import process_document
from app.features.document_extraction.schemas import DocumentType
from openpyxl import Workbook
from PIL import Image


async def collect_results(generator) -> list[dict]:
    return [result async for result in generator]


def create_pdf(file_path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Pipeline PDF text")
    document.save(file_path)
    document.close()


def create_xlsx(file_path: Path) -> None:
    workbook = Workbook()
    workbook.active.append(["name", "amount"])
    workbook.active.append(["Alice", 10])
    workbook.save(file_path)


def create_ocr() -> MagicMock:
    ocr = MagicMock()
    ocr.extract_image = AsyncMock(
        return_value={
            "text": "Pipeline image text",
            "confidence": 0.91,
            "words": [],
            "image_width": 20,
            "image_height": 20,
            "extraction_method": "google_vision",
        }
    )
    return ocr


@pytest.mark.integration
def test_pdf_pipeline_processes_real_pdf(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "document.pdf"
    create_pdf(file_path)

    results = asyncio.run(
        collect_results(
            process_document("document-01", file_path, DocumentType.PDF)
        )
    )

    assert len(results) == 1
    assert results[0]["kind"] == "page"
    assert results[0]["status"] == "completed"
    assert results[0]["text"] == "Pipeline PDF text"
    assert results[0]["metadata"]["total_pages"] == 1


@pytest.mark.integration
def test_csv_pipeline_processes_real_csv(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "data.csv"
    file_path.write_text("name,amount\nAlice,10\n", encoding="utf-8")

    results = asyncio.run(
        collect_results(
            process_document("document-02", file_path, DocumentType.CSV)
        )
    )

    assert len(results) == 1
    assert results[0]["kind"] == "csv_chunk"
    assert results[0]["status"] == "completed"
    assert results[0]["metadata"]["delimiter"] == ","
    assert results[0]["rows"][0]["cells"][0]["value"] == "name"


@pytest.mark.integration
def test_xlsx_pipeline_processes_real_workbook(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "data.xlsx"
    create_xlsx(file_path)

    results = asyncio.run(
        collect_results(
            process_document("document-03", file_path, DocumentType.XLSX)
        )
    )

    assert len(results) == 1
    assert results[0]["kind"] == "sheet_chunk"
    assert results[0]["status"] == "completed"
    assert results[0]["sheet_name"] == "Sheet"
    assert results[0]["rows"][1]["cells"][0]["value"] == "Alice"


@pytest.mark.integration
def test_image_pipeline_processes_real_image_with_mock_ocr(
    workspace_tmp_path: Path,
):
    file_path = workspace_tmp_path / "image.png"
    Image.new("RGB", (20, 20), color="white").save(file_path)
    ocr = create_ocr()

    results = asyncio.run(
        collect_results(
            process_document("document-04", file_path, DocumentType.PNG, ocr)
        )
    )

    assert len(results) == 1
    assert results[0]["kind"] == "page"
    assert results[0]["status"] == "completed"
    assert results[0]["text"] == "Pipeline image text"
    assert results[0]["metadata"]["confidence"] == 0.91
    ocr.extract_image.assert_awaited_once()
