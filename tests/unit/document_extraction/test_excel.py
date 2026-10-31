import math
from datetime import date, datetime, time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import xlrd
from app.features.document_extraction.data_processing.excel import (
    ExcelProcessingError,
    _safe_value,
    extract_excel_chunks,
)
from openpyxl import Workbook


def create_xlsx(file_path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Report"
    sheet.append(["Name", "Amount"])
    sheet.append(["Alice", 10])
    sheet.append(["Total", "=SUM(B2:B2)"])
    workbook.save(file_path)


def create_empty_xlsx(file_path: Path) -> None:
    workbook = Workbook()
    workbook.active.title = "Empty"
    workbook.save(file_path)


@pytest.mark.unit
def test_safe_value_converts_datetime_values():
    assert _safe_value(datetime(2026, 1, 2, 3, 4, 5)) == "2026-01-02T03:04:05"
    assert _safe_value(date(2026, 1, 2)) == "2026-01-02"
    assert _safe_value(time(3, 4, 5)) == "03:04:05"


@pytest.mark.unit
def test_safe_value_converts_non_finite_and_unknown_values():
    unknown_value = object()

    assert _safe_value(math.nan) == "nan"
    assert _safe_value(math.inf) == "inf"
    assert _safe_value(unknown_value) == str(unknown_value)
    assert _safe_value("text") == "text"
    assert _safe_value(10) == 10
    assert _safe_value(None) is None


@pytest.mark.unit
def test_extract_xlsx_chunks_preserves_rows_and_formulas(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "report.xlsx"
    create_xlsx(file_path)

    chunks = list(extract_excel_chunks(file_path, chunk_size=2))

    assert len(chunks) == 2
    assert chunks[0]["sheet_name"] == "Report"
    assert chunks[0]["start_row"] == 1
    assert chunks[0]["end_row"] == 2
    assert chunks[0]["rows"][0]["cells"][0]["value"] == "Name"
    assert chunks[1]["rows"][0]["cells"][1]["formula"] == "=SUM(B2:B2)"
    assert chunks[1]["rows"][0]["cells"][1]["value"] is None


@pytest.mark.unit
def test_extract_xlsx_emits_metadata_for_empty_sheet(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "empty.xlsx"
    create_empty_xlsx(file_path)

    chunks = list(extract_excel_chunks(file_path))

    assert chunks == [
        {
            "sheet_name": "Empty",
            "sheet_index": 1,
            "status": "completed",
            "start_row": None,
            "end_row": None,
            "rows": [],
            "error": None,
        }
    ]


@pytest.mark.unit
def test_extract_xlsx_rejects_column_limit(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "report.xlsx"
    create_xlsx(file_path)

    chunks = list(extract_excel_chunks(file_path, max_columns=1))

    assert chunks[0]["status"] == "failed"
    assert "column limit" in (chunks[0]["error"] or "")


@pytest.mark.unit
def test_extract_excel_rejects_missing_file(workspace_tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Excel file not found"):
        list(extract_excel_chunks(workspace_tmp_path / "missing.xlsx"))


@pytest.mark.unit
def test_extract_excel_rejects_unsupported_extension(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "report.csv"
    file_path.write_text("name,amount", encoding="utf-8")

    with pytest.raises(ExcelProcessingError, match="Supported formats"):
        list(extract_excel_chunks(file_path))


@pytest.mark.unit
def test_extract_excel_rejects_invalid_limits(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "report.xlsx"
    create_xlsx(file_path)

    with pytest.raises(ValueError, match="must be positive"):
        list(extract_excel_chunks(file_path, chunk_size=0))

    with pytest.raises(ValueError, match="must be positive"):
        list(extract_excel_chunks(file_path, max_columns=0))


@pytest.mark.unit
def test_extract_excel_wraps_corrupted_xlsx_error(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "corrupted.xlsx"
    file_path.write_bytes(b"not an xlsx")

    with pytest.raises(ExcelProcessingError, match="Unable to open or read"):
        list(extract_excel_chunks(file_path))


@pytest.mark.unit
def test_extract_xlsx_logs_sheet_failure(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "report.xlsx"
    create_xlsx(file_path)

    with (
        patch(
            "app.features.document_extraction.data_processing.excel._iter_xlsx_rows",
            side_effect=RuntimeError("row failure"),
        ),
        patch(
            "app.features.document_extraction.data_processing.excel.logger.exception"
        ) as log_exception,
    ):
        chunks = list(extract_excel_chunks(file_path))

    assert chunks[0]["status"] == "failed"
    log_exception.assert_called_once_with(
        "excel_sheet_processing_failed",
        file_path=str(file_path),
        sheet_name="Report",
        sheet_index=1,
        error="row failure",
    )


@pytest.mark.unit
def test_extract_xls_converts_common_cell_types(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "report.xls"
    file_path.write_bytes(b"legacy workbook placeholder")

    number_cell = MagicMock(ctype=xlrd.XL_CELL_NUMBER, value=12.5)
    boolean_cell = MagicMock(ctype=xlrd.XL_CELL_BOOLEAN, value=1)
    error_cell = MagicMock(ctype=xlrd.XL_CELL_ERROR, value=999)
    empty_cell = MagicMock(ctype=xlrd.XL_CELL_EMPTY, value="")
    sheet = MagicMock(
        ncols=4,
        nrows=1,
        row=MagicMock(return_value=[number_cell, boolean_cell, error_cell, empty_cell]),
    )
    book = MagicMock(
        datemode=0,
        sheet_names=MagicMock(return_value=["Legacy"]),
        sheet_by_index=MagicMock(return_value=sheet),
        sheet_loaded=MagicMock(return_value=False),
    )
    book.__enter__.return_value = book
    book.__exit__.return_value = None

    with patch(
        "app.features.document_extraction.data_processing.excel.xlrd.open_workbook",
        return_value=book,
    ):
        chunks = list(extract_excel_chunks(file_path))

    cells = chunks[0]["rows"][0]["cells"]
    assert cells[0]["value"] == 12.5
    assert cells[1]["value"] is True
    assert cells[2]["value"] == "#ERROR"
    assert len(cells) == 3
