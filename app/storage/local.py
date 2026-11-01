import os
import re
import shutil
from pathlib import Path

import aiofiles
from fastapi import UploadFile

BASE_DIR = Path(__file__).resolve().parents[2]

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads"))).resolve()

CHUNK_SIZE = 1024 * 1024
MAX_FILE_SIZE_MB = 50


class FileStorageError(ValueError):
    """Raised when local file storage validation fails."""


def get_document_directory(document_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", document_id):
        raise FileStorageError("Invalid document ID.")
    return UPLOAD_DIR / document_id


async def save_file(
    file: UploadFile,
    document_id: str,
    extension: str,
    max_size_mb: int = MAX_FILE_SIZE_MB,
) -> tuple[Path, int]:
    if not re.fullmatch(r"[a-z0-9]{1,10}", extension):
        raise FileStorageError("Invalid file extension.")
    if max_size_mb <= 0:
        raise FileStorageError("Invalid file size limit.")

    document_dir = get_document_directory(document_id)
    file_path = document_dir / f"source.{extension}"
    max_size_bytes = max_size_mb * 1024 * 1024
    document_dir.mkdir(parents=True, exist_ok=False)

    total_size = 0
    try:
        async with aiofiles.open(file_path, "xb") as output:
            while chunk := await file.read(CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > max_size_bytes:
                    raise FileStorageError(f"File exceeds {max_size_mb} MB limit")
                await output.write(chunk)
            if total_size == 0:
                raise FileStorageError("Uploaded file is empty.")
        return file_path, total_size
    except Exception:
        shutil.rmtree(document_dir, ignore_errors=True)
        raise


def get_file_path(document_id: str) -> Path:
    document_dir = get_document_directory(document_id)

    if document_dir.is_symlink():
        raise FileStorageError("Invalid storage directory.")
    if not document_dir.is_dir():
        raise FileNotFoundError("Document not found")
    files = [
        path for path in document_dir.glob("source.*") if path.is_file() and not path.is_symlink()
    ]
    if len(files) != 1:
        raise FileNotFoundError("Stored document is missing or invalid.")
    return files[0]


def delete_file(document_id: str) -> bool:
    document_dir = get_document_directory(document_id)

    if document_dir.is_symlink():
        raise FileStorageError("Invalid storage directory.")
    if not document_dir.exists():
        return False

    shutil.rmtree(document_dir)

    return True
