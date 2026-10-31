from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pymupdf
import pytest
from app.features.document_extraction.schemas import DocumentType
from app.features.document_extraction.validators import (
    DocumentValidationError,
    validate_document,
    validate_extension,
    validate_file_integrity,
    validate_file_signature,
    validate_file_size,
    vlidate_extension,
)
from openpyxl import Workbook
from PIL import Image


@pytest.mark.unit
def test_validate_extension_returns_document_type():
    assert validate_extension("invoice.PDF") is DocumentType.PDF
    assert validate_extension("report.xlsx") is DocumentType.XLSX
    assert validate_extension(r"C:\uploads\report.xlsx") is DocumentType.XLSX
    assert vlidate_extension("legacy.csv") is DocumentType.CSV


@pytest.mark.unit
@pytest.mark.parametrize("filename", [None, "", " ", "document", "document.doc"])
def test_validate_extension_rejects_invalid_filename(filename):
    with pytest.raises(DocumentValidationError):
        validate_extension(filename)


@pytest.mark.unit
def test_validate_extension_rejects_unknown_extension():
    with pytest.raises(DocumentValidationError, match="Unsupported file extension"):
        validate_extension("document.doc")


@pytest.mark.unit
def test_validate_document_accepts_csv(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "data.csv"
    file_path.write_text("name,score\nAlice,10\n", encoding="utf-8")

    document_type, file_size = validate_document(file_path, "data.csv")

    assert document_type is DocumentType.CSV
    assert file_size == file_path.stat().st_size


@pytest.mark.unit
def test_validate_document_accepts_xlsx(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "data.xlsx"
    workbook = Workbook()
    workbook.active["A1"] = "name"
    workbook.save(file_path)

    document_type, _ = validate_document(file_path, "data.xlsx")

    assert document_type is DocumentType.XLSX


@pytest.mark.unit
def test_validate_document_accepts_image(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "image.png"
    Image.new("RGB", (10, 10), color="white").save(file_path)

    document_type, _ = validate_document(file_path, "image.png")

    assert document_type is DocumentType.PNG


@pytest.mark.unit
def test_validate_document_rejects_mismatched_signature(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "fake.pdf"
    file_path.write_bytes(b"not a pdf")

    with pytest.raises(
        DocumentValidationError,
        match="does not match its extension",
    ):
        validate_document(file_path, "fake.pdf")


@pytest.mark.unit
def test_validate_file_size_rejects_missing_file(workspace_tmp_path: Path):
    with pytest.raises(DocumentValidationError, match="does not exist"):
        validate_file_size(workspace_tmp_path / "missing.pdf")


@pytest.mark.unit
def test_validate_file_size_rejects_empty_file(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "empty.pdf"
    file_path.touch()

    with pytest.raises(DocumentValidationError, match="cannot be empty"):
        validate_file_size(file_path)


@pytest.mark.unit
def test_validate_file_size_rejects_oversized_file(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "large.pdf"
    file_path.write_bytes(b"x" * 1024)

    with pytest.raises(DocumentValidationError, match="exceeds"):
        validate_file_size(file_path, max_size_mb=0.0001)


@pytest.mark.unit
def test_validate_file_size_rejects_invalid_limit(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "file.pdf"
    file_path.write_bytes(b"x")

    with pytest.raises(ValueError, match="greater than zero"):
        validate_file_size(file_path, max_size_mb=0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("document_type", "header"),
    [
        (DocumentType.PDF, b"%PDF-1.7"),
        (DocumentType.JPG, b"\xff\xd8\xff"),
        (DocumentType.PNG, b"\x89PNG\r\n\x1a\n"),
        (DocumentType.TIFF, b"II*\x00"),
        (DocumentType.XLS, b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"),
    ],
)
def test_validate_file_signature_accepts_known_signatures(
    workspace_tmp_path: Path,
    document_type: DocumentType,
    header: bytes,
):
    file_path = workspace_tmp_path / "document.bin"
    file_path.write_bytes(header)

    validate_file_signature(file_path, document_type)


@pytest.mark.unit
def test_validate_csv_accepts_utf16_bom(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "data.csv"
    file_path.write_bytes("\ufeffname,score\nAlice,10\n".encode("utf-16"))

    document_type, _ = validate_document(file_path, "data.csv")

    assert document_type is DocumentType.CSV


@pytest.mark.unit
def test_validate_csv_rejects_binary_data(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "data.csv"
    file_path.write_bytes(b"name\x00score")

    with pytest.raises(DocumentValidationError, match="binary"):
        validate_document(file_path, "data.csv")


@pytest.mark.unit
def test_validate_xlsx_rejects_missing_workbook_parts(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "invalid.xlsx"
    with ZipFile(file_path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")

    with pytest.raises(DocumentValidationError, match="workbook structure"):
        validate_document(file_path, "invalid.xlsx")


@pytest.mark.unit
def test_validate_xlsx_rejects_corrupted_zip(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "invalid.xlsx"
    file_path.write_bytes(b"PK\x03\x04not-a-zip")

    with pytest.raises(DocumentValidationError, match="Invalid XLSX"):
        validate_document(file_path, "invalid.xlsx")


@pytest.mark.unit
def test_validate_image_rejects_corrupted_file(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "corrupted.png"
    file_path.write_bytes(b"\x89PNG\r\n\x1a\ncorrupted")

    with pytest.raises(DocumentValidationError, match="Corrupted"):
        validate_document(file_path, "corrupted.png")


@pytest.mark.unit
def test_validate_pdf_rejects_empty_document(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "empty.pdf"
    file_path.write_bytes(b"%PDF-1.7")

    class EmptyPdf:
        needs_pass = False
        page_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    with patch(
        "app.features.document_extraction.validators.pymupdf.open",
        return_value=EmptyPdf(),
    ):
        with pytest.raises(DocumentValidationError, match="no pages"):
            validate_file_integrity(file_path, DocumentType.PDF)


@pytest.mark.unit
def test_validate_pdf_rejects_password_protected_document(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "protected.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(
        file_path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-password",
        user_pw="user-password",
    )
    document.close()

    with pytest.raises(DocumentValidationError, match="Password-protected"):
        validate_document(file_path, "protected.pdf")


@pytest.mark.unit
def test_validate_xls_rejects_corrupted_workbook(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "corrupted.xls"
    file_path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"invalid")

    with pytest.raises(DocumentValidationError, match="Corrupted"):
        validate_document(file_path, "corrupted.xls")
