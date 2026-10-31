from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from app.features.document_extraction.data_processing.pdf import (
    PDFProcessingError,
    calculate_image_coverage,
    classify_pdf_page,
    extract_pdf_pages,
    get_pdf_page_count,
)


def create_pdf(file_path: Path, page_count: int = 1) -> None:
    document = pymupdf.open()
    for page_number in range(page_count):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {page_number + 1} text")
    document.save(file_path)
    document.close()


@pytest.mark.unit
def test_get_pdf_page_count_returns_page_count(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "document.pdf"
    create_pdf(file_path, page_count=2)

    assert get_pdf_page_count(file_path) == 2


@pytest.mark.unit
def test_get_pdf_page_count_rejects_missing_file(workspace_tmp_path: Path):
    with pytest.raises(PDFProcessingError, match="Unable to read PDF"):
        get_pdf_page_count(workspace_tmp_path / "missing.pdf")


@pytest.mark.unit
def test_get_pdf_page_count_rejects_corrupted_file(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "corrupted.pdf"
    file_path.write_bytes(b"not a PDF")

    with pytest.raises(PDFProcessingError, match="Unable to read PDF"):
        get_pdf_page_count(file_path)


@pytest.mark.unit
def test_calculate_image_coverage_returns_zero_for_text_page(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "document.pdf"
    create_pdf(file_path)

    with pymupdf.open(file_path) as document:
        assert calculate_image_coverage(document[0]) == 0.0


@pytest.mark.unit
def test_classify_pdf_page_returns_expected_types():
    page = MagicMock()
    page.get_drawings.return_value = []

    assert classify_pdf_page(page, "text", 0.0) == ("digital", False)
    assert classify_pdf_page(page, "text", 0.25) == ("mixed", True)
    assert classify_pdf_page(page, "", 0.5) == ("scanned", True)
    assert classify_pdf_page(page, "", 0.0) == ("blank", False)

    page.get_drawings.return_value = [{}]
    assert classify_pdf_page(page, "", 0.0) == ("unknown", True)


@pytest.mark.unit
def test_extract_pdf_pages_returns_text_and_metadata(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "document.pdf"
    create_pdf(file_path, page_count=2)

    pages = list(extract_pdf_pages(file_path, start_page=2, end_page=2))

    assert len(pages) == 1
    assert pages[0]["page_number"] == 2
    assert pages[0]["page_type"] == "digital"
    assert pages[0]["extraction_method"] == "pymupdf"
    assert pages[0]["text"] == "Page 2 text"
    assert pages[0]["needs_ocr"] is False
    assert pages[0]["error"] is None
    assert pages[0]["text_blocks"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("start_page", "end_page"),
    [(0, None), (2, 1), (1, 3)],
)
def test_extract_pdf_pages_rejects_invalid_range(
    workspace_tmp_path: Path,
    start_page: int,
    end_page: int | None,
):
    file_path = workspace_tmp_path / "document.pdf"
    create_pdf(file_path, page_count=2)

    with pytest.raises(ValueError, match="Invalid PDF page range"):
        list(extract_pdf_pages(file_path, start_page, end_page))


@pytest.mark.unit
def test_extract_pdf_pages_rejects_missing_file(workspace_tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        list(extract_pdf_pages(workspace_tmp_path / "missing.pdf"))


@pytest.mark.unit
def test_extract_pdf_pages_returns_error_result_and_logs_failure(
    workspace_tmp_path: Path,
):
    file_path = workspace_tmp_path / "document.pdf"
    create_pdf(file_path)

    with (
        patch(
            "app.features.document_extraction.data_processing.pdf.calculate_image_coverage",
            side_effect=RuntimeError("page processing failed"),
        ),
        patch(
            "app.features.document_extraction.data_processing.pdf.logger.exception"
        ) as log_exception,
    ):
        pages = list(extract_pdf_pages(file_path))

    assert pages[0]["page_type"] == "error"
    assert pages[0]["text"] == ""
    assert pages[0]["error"] == "Page extraction failed: RuntimeError"
    log_exception.assert_called_once_with(
        "pdf_page_processing_failed",
        page_number=1,
        error="page processing failed",
    )
