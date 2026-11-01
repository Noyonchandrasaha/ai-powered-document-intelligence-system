import codecs
import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict

from app.core.logging import get_logger
from openpyxl.utils import get_column_letter

logger = get_logger(__name__)


SAMPLE_BYTES = 64 * 1024
SAMPLE_CHARS = 8192

MAX_CHUNK_CELLS = 3000
MAX_CHUNK_TEXT_BYTES = 2 * 1024 * 1024


class CSVProcessingError(Exception):
    """Raised when a CSV document cannot be processed."""


class CSVChunk(TypedDict):
    encoding: str
    delimiter: str
    status: str
    start_row: int | None
    end_row: int | None
    rows: list[dict[str, Any]]
    error: str | None


def detect_csv_encoding(path: Path) -> str:
    """Detect common CSV encodings from a small sample."""

    with path.open("rb") as stream:
        sample = stream.read(SAMPLE_BYTES)

    if not sample:
        raise CSVProcessingError("CSV file is empty.")

    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"

    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    if b"\x00" in sample:
        raise CSVProcessingError("Binary data or UTF-16 without BOM; specify encoding explicitly.")

    try:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")

        decoder.decode(sample, final=False)

        return "utf-8"

    except UnicodeDecodeError:
        try:
            sample.decode("cp1252")
            return "cp1252"

        except UnicodeDecodeError as exc:
            raise CSVProcessingError("Unknown encoding; specify it explicitly.") from exc


def _build_chunk(
    rows: list[dict[str, Any]],
    encoding: str,
    delimiter: str,
) -> CSVChunk:
    """Build a normalized CSV result chunk."""

    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "status": "completed",
        "start_row": rows[0]["row_number"],
        "end_row": rows[-1]["row_number"],
        "rows": rows,
        "error": None,
    }


def extract_csv_chunks(
    file_path: str | Path,
    *,
    chunk_size: int = 200,
    encoding: str | None = None,
    delimiter: str | None = None,
    max_columns: int = 2000,
) -> Iterator[CSVChunk]:
    """
    Extract CSV records without loading the entire file.

    - Detects common encodings and delimiters.
    - Preserves raw string values and cell positions.
    - Supports quoted and multiline fields.
    - Returns bounded result chunks.
    - Reports malformed records.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if path.suffix.lower() != ".csv":
        raise CSVProcessingError("Expected a .csv file.")

    if chunk_size < 1 or max_columns < 1:
        raise ValueError("chunk_size and max_columns must be positive.")

    if delimiter is not None and (len(delimiter) != 1 or delimiter in '\r\n"'):
        raise ValueError("Invalid CSV delimiter.")

    try:
        selected_encoding = encoding or detect_csv_encoding(path)
    except CSVProcessingError as exc:
        logger.error(
            "csv_encoding_detection_failed",
            file_path=str(path),
            error=str(exc),
        )
        raise
    except (LookupError, OSError, UnicodeError) as exc:
        logger.exception(
            "csv_encoding_detection_failed",
            file_path=str(path),
            error=str(exc),
        )
        raise CSVProcessingError(f"Unable to detect CSV encoding: {type(exc).__name__}") from exc

    try:
        with path.open(
            "r",
            encoding=selected_encoding,
            errors="strict",
            newline="",
        ) as stream:
            sample = stream.read(SAMPLE_CHARS)

            if not sample.strip():
                raise CSVProcessingError("CSV contains no content.")

            stream.seek(0)

            # Detect CSV delimiter and quoting rules.
            if delimiter is None:
                try:
                    dialect = csv.Sniffer().sniff(
                        sample,
                        delimiters=",;\t|",
                    )

                    selected_delimiter = dialect.delimiter

                    reader = csv.reader(
                        stream,
                        dialect=dialect,
                        strict=True,
                    )

                except csv.Error:
                    selected_delimiter = ","

                    reader = csv.reader(
                        stream,
                        delimiter=",",
                        strict=True,
                    )

            else:
                selected_delimiter = delimiter

                reader = csv.reader(
                    stream,
                    delimiter=delimiter,
                    strict=True,
                )

            batch: list[dict[str, Any]] = []

            cell_count = 0
            text_bytes = 0
            expected_columns: int | None = None
            record_number = 0
            completed_chunks = 0

            try:
                for values in reader:
                    record_number += 1

                    if not values:
                        continue

                    if len(values) > max_columns:
                        raise CSVProcessingError(
                            f"Row {record_number} has "
                            f"{len(values)} columns; "
                            f"limit is {max_columns}."
                        )

                    row_bytes = sum(len(value.encode("utf-8")) for value in values)

                    if row_bytes > MAX_CHUNK_TEXT_BYTES:
                        raise CSVProcessingError(f"Row {record_number} exceeds result size limit.")

                    # Flush before exceeding chunk limits.
                    if batch and (
                        len(batch) >= chunk_size
                        or cell_count + len(values) > MAX_CHUNK_CELLS
                        or text_bytes + row_bytes > MAX_CHUNK_TEXT_BYTES
                    ):
                        yield _build_chunk(
                            batch,
                            selected_encoding,
                            selected_delimiter,
                        )

                        completed_chunks += 1
                        batch = []
                        cell_count = 0
                        text_bytes = 0

                    if expected_columns is None:
                        expected_columns = len(values)

                    warnings = []

                    if len(values) != expected_columns:
                        warnings.append(
                            f"Expected {expected_columns} columns; found {len(values)}."
                        )

                    cells = [
                        {
                            "coordinate": (f"{get_column_letter(column)}{record_number}"),
                            "value": value,
                            "data_type": "s",
                            "formula": None,
                            "number_format": None,
                        }
                        for column, value in enumerate(
                            values,
                            start=1,
                        )
                    ]

                    batch.append(
                        {
                            "row_number": record_number,
                            "line_number": reader.line_num,
                            "cells": cells,
                            "warnings": warnings,
                        }
                    )

                    cell_count += len(values)
                    text_bytes += row_bytes

                if batch:
                    yield _build_chunk(
                        batch,
                        selected_encoding,
                        selected_delimiter,
                    )

                    completed_chunks += 1

                if completed_chunks == 0:
                    raise CSVProcessingError("CSV contains no data records.")

            except (
                csv.Error,
                UnicodeError,
                CSVProcessingError,
            ) as exc:
                logger.exception(
                    "csv_parsing_failed",
                    file_path=str(path),
                    line_number=reader.line_num,
                    error=str(exc),
                )

                # Preserve successfully parsed rows.
                if batch:
                    yield _build_chunk(
                        batch,
                        selected_encoding,
                        selected_delimiter,
                    )

                yield {
                    "encoding": selected_encoding,
                    "delimiter": selected_delimiter,
                    "status": "failed",
                    "start_row": None,
                    "end_row": None,
                    "rows": [],
                    "error": (f"CSV parsing failed near line {reader.line_num}: {exc}"),
                }

    except (LookupError, OSError, UnicodeError) as exc:
        logger.exception(
            "csv_open_failed",
            file_path=str(path),
            error=str(exc),
        )
        raise CSVProcessingError(f"Unable to open CSV: {type(exc).__name__}") from exc
