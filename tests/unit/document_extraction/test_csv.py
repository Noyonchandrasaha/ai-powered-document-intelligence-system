from pathlib import Path
from unittest.mock import patch

import pytest
from app.features.document_extraction.data_processing.csv import (
    CSVProcessingError,
    detect_csv_encoding,
    extract_csv_chunks,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("content", "expected_encoding"),
    [
        (b"name,amount\nAlice,10\n", "utf-8"),
        (b"\xef\xbb\xbfname,amount\nAlice,10\n", "utf-8-sig"),
        ("name,amount\nAlice,10\n".encode("utf-16"), "utf-16"),
        ("name,amount\nCaf\u00e9,10\n".encode("cp1252"), "cp1252"),
    ],
)
def test_detect_csv_encoding_returns_expected_encoding(
    workspace_tmp_path: Path,
    content: bytes,
    expected_encoding: str,
):
    file_path = workspace_tmp_path / "data.csv"
    file_path.write_bytes(content)

    assert detect_csv_encoding(file_path) == expected_encoding


@pytest.mark.unit
def test_detect_csv_encoding_rejects_empty_file(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "empty.csv"
    file_path.touch()

    with pytest.raises(CSVProcessingError, match="file is empty"):
        detect_csv_encoding(file_path)


@pytest.mark.unit
def test_detect_csv_encoding_rejects_binary_data(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "binary.csv"
    file_path.write_bytes(b"name\x00amount")

    with pytest.raises(CSVProcessingError, match="Binary data"):
        detect_csv_encoding(file_path)


@pytest.mark.unit
def test_extract_csv_chunks_preserves_cells_and_chunks_rows(
    workspace_tmp_path: Path,
):
    file_path = workspace_tmp_path / "data.csv"
    file_path.write_text(
        "name,amount\nAlice,10\nBob,20\n",
        encoding="utf-8",
    )

    chunks = list(extract_csv_chunks(file_path, chunk_size=2))

    assert len(chunks) == 2
    assert chunks[0]["encoding"] == "utf-8"
    assert chunks[0]["delimiter"] == ","
    assert chunks[0]["start_row"] == 1
    assert chunks[0]["end_row"] == 2
    assert chunks[0]["rows"][0]["cells"][0]["coordinate"] == "A1"
    assert chunks[0]["rows"][1]["cells"][1]["value"] == "10"
    assert chunks[1]["rows"][0]["line_number"] == 3


@pytest.mark.unit
def test_extract_csv_chunks_detects_semicolon_delimiter(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "data.csv"
    file_path.write_text("name;amount\nAlice;10\n", encoding="utf-8")

    chunks = list(extract_csv_chunks(file_path))

    assert chunks[0]["delimiter"] == ";"
    assert chunks[0]["rows"][1]["cells"][1]["value"] == "10"


@pytest.mark.unit
def test_extract_csv_chunks_supports_explicit_encoding_and_delimiter(
    workspace_tmp_path: Path,
):
    file_path = workspace_tmp_path / "data.csv"
    file_path.write_text("name|amount\nCafé|10\n", encoding="cp1252")

    chunks = list(
        extract_csv_chunks(
            file_path,
            encoding="cp1252",
            delimiter="|",
        )
    )

    assert chunks[0]["encoding"] == "cp1252"
    assert chunks[0]["delimiter"] == "|"
    assert chunks[0]["rows"][1]["cells"][0]["value"] == "Café"


@pytest.mark.unit
def test_extract_csv_chunks_supports_quoted_multiline_values(
    workspace_tmp_path: Path,
):
    file_path = workspace_tmp_path / "data.csv"
    file_path.write_text(
        'name,notes\nAlice,"line one\nline two"\n',
        encoding="utf-8",
    )

    chunks = list(extract_csv_chunks(file_path))

    multiline_value = chunks[0]["rows"][1]["cells"][1]["value"]
    assert multiline_value.replace("\r\n", "\n") == "line one\nline two"
    assert chunks[0]["rows"][1]["line_number"] == 3


@pytest.mark.unit
def test_extract_csv_chunks_reports_mismatched_columns(
    workspace_tmp_path: Path,
):
    file_path = workspace_tmp_path / "data.csv"
    file_path.write_text("name,amount\nAlice,10,extra\n", encoding="utf-8")

    chunks = list(extract_csv_chunks(file_path))

    warnings = chunks[0]["rows"][1]["warnings"]
    assert warnings == ["Expected 2 columns; found 3."]


@pytest.mark.unit
def test_extract_csv_chunks_reports_malformed_records_and_logs(
    workspace_tmp_path: Path,
):
    file_path = workspace_tmp_path / "data.csv"
    file_path.write_text('name,amount\n"Alice,10\n', encoding="utf-8")

    with patch(
        "app.features.document_extraction.data_processing.csv.logger.exception"
    ) as log_exception:
        chunks = list(extract_csv_chunks(file_path))

    assert any(chunk["status"] == "failed" for chunk in chunks)
    log_exception.assert_called_once()
    assert log_exception.call_args.args[0] == "csv_parsing_failed"


@pytest.mark.unit
def test_extract_csv_chunks_rejects_column_limit(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "data.csv"
    file_path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    chunks = list(extract_csv_chunks(file_path, max_columns=2))

    assert chunks[0]["status"] == "failed"
    assert "limit is 2" in (chunks[0]["error"] or "")


@pytest.mark.unit
def test_extract_csv_chunks_rejects_invalid_input(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "data.csv"
    file_path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be positive"):
        list(extract_csv_chunks(file_path, chunk_size=0))

    with pytest.raises(ValueError, match="Invalid CSV delimiter"):
        list(extract_csv_chunks(file_path, delimiter="||"))


@pytest.mark.unit
def test_extract_csv_chunks_rejects_missing_and_wrong_extension(
    workspace_tmp_path: Path,
):
    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        list(extract_csv_chunks(workspace_tmp_path / "missing.csv"))

    file_path = workspace_tmp_path / "data.txt"
    file_path.write_text("a,b", encoding="utf-8")

    with pytest.raises(CSVProcessingError, match="Expected a .csv"):
        list(extract_csv_chunks(file_path))
