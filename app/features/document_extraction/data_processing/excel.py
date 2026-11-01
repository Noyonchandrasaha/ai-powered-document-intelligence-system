import math
from collections.abc import Iterator
from datetime import date, datetime, time
from itertools import zip_longest
from pathlib import Path
from typing import Any, TypedDict

import xlrd
from app.core.logging import get_logger
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

logger = get_logger(__name__)


class ExcelProcessingError(Exception):
    """Raised when an Excel file cannot be processed."""


class ExcelChunk(TypedDict):
    sheet_name: str
    sheet_index: int
    status: str
    start_row: int | None
    end_row: int | None
    rows: list[dict[str, Any]]
    error: str | None


def _safe_value(value: Any) -> Any:
    """Convert Excel values to JSON-compatible values."""

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    if isinstance(value, float) and not math.isfinite(value):
        return str(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _as_chunks(
    rows: Iterator[dict[str, Any]],
    sheet_name: str,
    sheet_index: int,
    chunk_size: int,
) -> Iterator[ExcelChunk]:
    """Produce small batches of non-empty worksheet rows."""

    batch: list[dict[str, Any]] = []
    emitted = False

    for row in rows:
        if not row["cells"]:
            continue

        batch.append(row)

        if len(batch) >= chunk_size:
            emitted = True

            yield {
                "sheet_name": sheet_name,
                "sheet_index": sheet_index,
                "status": "completed",
                "start_row": batch[0]["row_number"],
                "end_row": batch[-1]["row_number"],
                "rows": batch,
                "error": None,
            }

            batch = []

    # Emit any remaining rows, including empty-sheet metadata.
    if batch or not emitted:
        yield {
            "sheet_name": sheet_name,
            "sheet_index": sheet_index,
            "status": "completed",
            "start_row": (batch[0]["row_number"] if batch else None),
            "end_row": (batch[-1]["row_number"] if batch else None),
            "rows": batch,
            "error": None,
        }


def _iter_xlsx_rows(
    raw_sheet: Any,
    value_sheet: Any,
) -> Iterator[dict[str, Any]]:
    """Read XLSX cells while preserving formulas and cached values."""

    raw_rows = raw_sheet.iter_rows()
    value_rows = value_sheet.iter_rows()

    for row_number, (source_row, cached_row) in enumerate(
        zip_longest(
            raw_rows,
            value_rows,
            fillvalue=(),
        ),
        start=1,
    ):
        cached_cells = {cell.coordinate: cell for cell in cached_row if cell.value is not None}

        cells = []

        for cell in source_row:
            if cell.value is None:
                continue

            is_formula = cell.data_type == "f"

            cached_cell = cached_cells.get(cell.coordinate)

            if is_formula:
                value = cached_cell.value if cached_cell is not None else None
            else:
                value = cell.value

            cells.append(
                {
                    "coordinate": cell.coordinate,
                    "value": _safe_value(value),
                    "data_type": cell.data_type,
                    "formula": (str(cell.value) if is_formula else None),
                    "number_format": cell.number_format,
                }
            )

        yield {
            "row_number": row_number,
            "cells": cells,
        }


def _extract_xlsx(
    path: Path,
    chunk_size: int,
    max_columns: int,
) -> Iterator[ExcelChunk]:
    """Process XLSX sheets using read-only workbooks."""

    raw_book = load_workbook(
        path,
        read_only=True,
        data_only=False,
        keep_links=False,
    )

    try:
        value_book = load_workbook(
            path,
            read_only=True,
            data_only=True,
            keep_links=False,
        )
    except Exception:
        raw_book.close()
        raise

    try:
        for index, name in enumerate(
            raw_book.sheetnames,
            start=1,
        ):
            try:
                raw_sheet = raw_book[name]
                value_sheet = value_book[name]

                if raw_sheet.max_column and raw_sheet.max_column > max_columns:
                    raise ExcelProcessingError("Sheet exceeds configured column limit.")

                yield from _as_chunks(
                    _iter_xlsx_rows(
                        raw_sheet,
                        value_sheet,
                    ),
                    name,
                    index,
                    chunk_size,
                )

            except Exception as exc:
                logger.exception(
                    "excel_sheet_processing_failed",
                    file_path=str(path),
                    sheet_name=name,
                    sheet_index=index,
                    error=str(exc),
                )

                yield {
                    "sheet_name": name,
                    "sheet_index": index,
                    "status": "failed",
                    "start_row": None,
                    "end_row": None,
                    "rows": [],
                    "error": (f"{type(exc).__name__}: {exc}"),
                }

    finally:
        raw_book.close()
        value_book.close()


def _extract_xls(
    path: Path,
    chunk_size: int,
    max_columns: int,
) -> Iterator[ExcelChunk]:
    """Process legacy XLS files sheet by sheet."""

    with xlrd.open_workbook(
        str(path),
        on_demand=True,
    ) as book:
        for index, name in enumerate(
            book.sheet_names(),
            start=1,
        ):
            try:
                sheet = book.sheet_by_index(index - 1)

                if sheet.ncols > max_columns:
                    raise ExcelProcessingError("Sheet exceeds configured column limit.")

                def iter_rows(current_sheet: Any = sheet) -> Iterator[dict[str, Any]]:
                    for row_idx in range(current_sheet.nrows):
                        cells = []

                        for col_idx, cell in enumerate(current_sheet.row(row_idx)):
                            if cell.ctype in (
                                xlrd.XL_CELL_EMPTY,
                                xlrd.XL_CELL_BLANK,
                            ):
                                continue

                            value: Any

                            if cell.ctype == xlrd.XL_CELL_DATE:
                                value = xlrd.xldate_as_datetime(
                                    float(cell.value),
                                    book.datemode,
                                )

                            elif cell.ctype == xlrd.XL_CELL_ERROR:
                                value = xlrd.error_text_from_code.get(
                                    int(cell.value),
                                    "#ERROR",
                                )

                            elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
                                value = bool(cell.value)

                            else:
                                value = cell.value

                            cells.append(
                                {
                                    "coordinate": (
                                        f"{get_column_letter(col_idx + 1)}{row_idx + 1}"
                                    ),
                                    "value": _safe_value(value),
                                    "data_type": str(cell.ctype),
                                    "formula": None,
                                    "number_format": None,
                                }
                            )

                        yield {
                            "row_number": row_idx + 1,
                            "cells": cells,
                        }

                yield from _as_chunks(
                    iter_rows(),
                    name,
                    index,
                    chunk_size,
                )

            except Exception as exc:
                logger.exception(
                    "excel_sheet_processing_failed",
                    file_path=str(path),
                    sheet_name=name,
                    sheet_index=index,
                    error=str(exc),
                )

                yield {
                    "sheet_name": name,
                    "sheet_index": index,
                    "status": "failed",
                    "start_row": None,
                    "end_row": None,
                    "rows": [],
                    "error": (f"{type(exc).__name__}: {exc}"),
                }

            finally:
                if book.sheet_loaded(index - 1):
                    book.unload_sheet(index - 1)


def extract_excel_chunks(
    file_path: str | Path,
    *,
    chunk_size: int = 200,
    max_columns: int = 2000,
) -> Iterator[ExcelChunk]:
    """
    Extract XLSX/XLS data sheet by sheet.

    - Yields small result chunks.
    - Preserves original row numbers and cell coordinates.
    - Does not calculate Excel formulas.
    - Reports sheet-level extraction failures.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Excel file not found: {path}")

    if path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ExcelProcessingError("Supported formats are .xlsx and .xls.")

    if chunk_size <= 0 or max_columns <= 0:
        raise ValueError("chunk_size and max_columns must be positive.")

    try:
        if path.suffix.lower() == ".xlsx":
            yield from _extract_xlsx(
                path,
                chunk_size,
                max_columns,
            )
        else:
            yield from _extract_xls(
                path,
                chunk_size,
                max_columns,
            )

    except Exception as exc:
        logger.exception(
            "excel_processing_failed",
            file_path=str(path),
            error=str(exc),
        )
        raise ExcelProcessingError("Unable to open or read Excel file.") from exc
