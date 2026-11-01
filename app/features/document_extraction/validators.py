from pathlib import Path
from zipfile import BadZipFile, ZipFile, is_zipfile

import pymupdf
import xlrd
from app.features.document_extraction.schemas import DocumentType
from openpyxl import load_workbook
from PIL import Image

MAX_FILE_SIZE_MB = 50
MAX_UNCOMPRESSED_ARCHIVE_SIZE = 200 * 1024 * 1024

SUPPORTED_EXTENSIONS = {
    ".pdf": DocumentType.PDF,
    ".jpg": DocumentType.JPG,
    ".jpeg": DocumentType.JPG,
    ".png": DocumentType.PNG,
    ".tif": DocumentType.TIFF,
    ".tiff": DocumentType.TIFF,
    ".xlsx": DocumentType.XLSX,
    ".xls": DocumentType.XLS,
    ".csv": DocumentType.CSV,
}


class DocumentValidationError(ValueError):
    """Raised when an uploaded document is invalid or unreadable."""


def validate_extension(filename: str | None) -> DocumentType:
    if not filename or not filename.strip():
        raise DocumentValidationError("Filename is required.")

    normalized_name = filename.replace("\\", "/").split("/")[-1]
    extension = Path(normalized_name).suffix.lower()
    document_type = SUPPORTED_EXTENSIONS.get(extension)

    if document_type is None:
        raise DocumentValidationError(f"Unsupported file extension: {extension or 'none'}")
    return document_type


# Backward-compatible alias for callers using the original misspelling.
vlidate_extension = validate_extension


def validate_file_size(
    file_path: Path,
    max_size_mb: int = MAX_FILE_SIZE_MB,
) -> int:
    if max_size_mb <= 0:
        raise ValueError("max_size_mb must be greater than zero.")
    if not file_path.is_file():
        raise DocumentValidationError("File does not exist.")

    file_size = file_path.stat().st_size
    max_size = max_size_mb * 1024 * 1024
    if file_size == 0:
        raise DocumentValidationError("File cannot be empty.")
    if file_size > max_size:
        raise DocumentValidationError(f"File exceeds the {max_size_mb} MB limit.")
    return file_size


def _validate_xlsx_archive(file_path: Path) -> None:
    if not is_zipfile(file_path):
        raise DocumentValidationError("Invalid XLSX file structure.")

    try:
        with ZipFile(file_path) as archive:
            infos = archive.infolist()
            uncompressed_size = sum(info.file_size for info in infos)
            if uncompressed_size > MAX_UNCOMPRESSED_ARCHIVE_SIZE:
                raise DocumentValidationError("XLSX archive expands beyond the safe limit.")

            names = {info.filename for info in infos}
            if "[Content_Types].xml" not in names or "xl/workbook.xml" not in names:
                raise DocumentValidationError("Invalid Excel workbook structure.")

            bad_member = archive.testzip()
            if bad_member is not None:
                raise DocumentValidationError("XLSX archive contains corrupted data.")
    except BadZipFile as exc:
        raise DocumentValidationError("Invalid XLSX file structure.") from exc


def validate_file_signature(file_path: Path, document_type: DocumentType) -> None:
    try:
        with file_path.open("rb") as file:
            header = file.read(1024)
    except OSError as exc:
        raise DocumentValidationError("Unable to read the uploaded file.") from exc

    if document_type == DocumentType.CSV:
        is_utf16 = header.startswith((b"\xff\xfe", b"\xfe\xff"))
        if b"\x00" in header and not is_utf16:
            raise DocumentValidationError("CSV appears to contain binary data.")
        return

    signatures = {
        DocumentType.PDF: (b"%PDF-",),
        DocumentType.JPG: (b"\xff\xd8\xff",),
        DocumentType.PNG: (b"\x89PNG\r\n\x1a\n",),
        DocumentType.TIFF: (b"II*\x00", b"MM\x00*"),
        DocumentType.XLS: (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),
        DocumentType.XLSX: (b"PK\x03\x04",),
    }
    expected = signatures.get(document_type, ())

    if not any(header.startswith(signature) for signature in expected):
        raise DocumentValidationError("File content does not match its extension.")

    if document_type == DocumentType.XLSX:
        _validate_xlsx_archive(file_path)


def validate_file_integrity(file_path: Path, document_type: DocumentType) -> None:
    try:
        if document_type == DocumentType.PDF:
            with pymupdf.open(file_path) as document:  # type: ignore[no-untyped-call]
                if document.needs_pass:
                    raise DocumentValidationError("Password-protected PDFs are not supported.")
                if document.page_count == 0:
                    raise DocumentValidationError("PDF contains no pages.")
        elif document_type in {
            DocumentType.JPG,
            DocumentType.PNG,
            DocumentType.TIFF,
        }:
            with Image.open(file_path) as image:
                image.verify()
        elif document_type == DocumentType.XLSX:
            workbook = load_workbook(file_path, read_only=True, keep_links=False)
            try:
                if not workbook.sheetnames:
                    raise DocumentValidationError("Excel workbook contains no sheets.")
            finally:
                workbook.close()
        elif document_type == DocumentType.XLS:
            xls_workbook: object = xlrd.open_workbook(str(file_path), on_demand=True)
            try:
                if xls_workbook.nsheets == 0:  # type: ignore[attr-defined]
                    raise DocumentValidationError("Excel workbook contains no sheets.")
            finally:
                xls_workbook.release_resources()  # type: ignore[attr-defined]
    except DocumentValidationError:
        raise
    except Exception as exc:
        raise DocumentValidationError(
            f"Corrupted or unreadable {document_type.value} file."
        ) from exc


def validate_document(
    file_path: Path,
    original_filename: str,
    max_size_mb: int = MAX_FILE_SIZE_MB,
) -> tuple[DocumentType, int]:
    document_type = validate_extension(original_filename)
    file_size = validate_file_size(file_path, max_size_mb=max_size_mb)
    validate_file_signature(file_path, document_type)
    validate_file_integrity(file_path, document_type)
    return document_type, file_size
