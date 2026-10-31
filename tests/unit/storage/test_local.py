import asyncio
import io
from pathlib import Path
from unittest.mock import patch

import pytest
from app.storage import local
from fastapi import UploadFile


def create_upload_file(content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename="document.pdf")


@pytest.fixture
def upload_dir(workspace_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = workspace_tmp_path / "uploads"
    path.mkdir()
    monkeypatch.setattr(local, "UPLOAD_DIR", path)
    return path


@pytest.mark.unit
@pytest.mark.parametrize("document_id", ["", "../escape", "document/id", "document id", "a" * 65])
def test_get_document_directory_rejects_invalid_id(document_id: str):
    with pytest.raises(local.FileStorageError, match="Invalid document ID"):
        local.get_document_directory(document_id)


@pytest.mark.unit
def test_get_document_directory_returns_document_path(upload_dir: Path):
    assert local.get_document_directory("document-01") == upload_dir / "document-01"


@pytest.mark.unit
@pytest.mark.parametrize("extension", ["", "PDF", "pdf.", "pdf/", "p" * 11])
def test_save_file_rejects_invalid_extension(upload_dir: Path, extension: str):
    with pytest.raises(local.FileStorageError, match="Invalid file extension"):
        asyncio.run(local.save_file(create_upload_file(b"content"), "document-01", extension))


@pytest.mark.unit
@pytest.mark.parametrize("max_size_mb", [0, -1])
def test_save_file_rejects_invalid_size_limit(upload_dir: Path, max_size_mb: int):
    with pytest.raises(local.FileStorageError, match="Invalid file size limit"):
        asyncio.run(
            local.save_file(
                create_upload_file(b"content"),
                "document-01",
                "pdf",
                max_size_mb,
            )
        )


@pytest.mark.unit
def test_save_file_saves_upload(upload_dir: Path):
    content = b"document content"

    file_path, file_size = asyncio.run(
        local.save_file(create_upload_file(content), "document-01", "pdf")
    )

    assert file_path == upload_dir / "document-01" / "source.pdf"
    assert file_path.read_bytes() == content
    assert file_size == len(content)


@pytest.mark.unit
def test_save_file_rejects_empty_upload_and_cleans_directory(upload_dir: Path):
    with pytest.raises(local.FileStorageError, match="empty"):
        asyncio.run(local.save_file(create_upload_file(b""), "document-01", "pdf"))

    assert not (upload_dir / "document-01").exists()


@pytest.mark.unit
def test_save_file_rejects_oversized_upload_and_cleans_directory(upload_dir: Path):
    content = b"x" * (1024 * 1024 + 1)

    with pytest.raises(local.FileStorageError, match="exceeds"):
        asyncio.run(local.save_file(create_upload_file(content), "document-01", "pdf", 1))

    assert not (upload_dir / "document-01").exists()


@pytest.mark.unit
def test_save_file_rejects_duplicate_document(upload_dir: Path):
    asyncio.run(local.save_file(create_upload_file(b"first"), "document-01", "pdf"))

    with pytest.raises(FileExistsError):
        asyncio.run(local.save_file(create_upload_file(b"second"), "document-01", "pdf"))


@pytest.mark.unit
def test_get_file_path_returns_stored_file(upload_dir: Path):
    document_dir = upload_dir / "document-01"
    document_dir.mkdir()
    file_path = document_dir / "source.pdf"
    file_path.write_bytes(b"content")

    assert local.get_file_path("document-01") == file_path


@pytest.mark.unit
def test_get_file_path_rejects_missing_document(upload_dir: Path):
    with pytest.raises(FileNotFoundError, match="Document not found"):
        local.get_file_path("document-01")


@pytest.mark.unit
def test_get_file_path_rejects_missing_stored_file(upload_dir: Path):
    (upload_dir / "document-01").mkdir()

    with pytest.raises(FileNotFoundError, match="missing or invalid"):
        local.get_file_path("document-01")


@pytest.mark.unit
def test_get_file_path_rejects_multiple_stored_files(upload_dir: Path):
    document_dir = upload_dir / "document-01"
    document_dir.mkdir()
    (document_dir / "source.pdf").write_bytes(b"pdf")
    (document_dir / "source.txt").write_bytes(b"text")

    with pytest.raises(FileNotFoundError, match="missing or invalid"):
        local.get_file_path("document-01")


@pytest.mark.unit
def test_get_file_path_rejects_symlink_document_directory(upload_dir: Path):
    (upload_dir / "document-01").mkdir()

    with patch.object(Path, "is_symlink", return_value=True):
        with pytest.raises(local.FileStorageError, match="Invalid storage directory"):
            local.get_file_path("document-01")


@pytest.mark.unit
def test_delete_file_returns_false_for_missing_document(upload_dir: Path):
    assert local.delete_file("document-01") is False


@pytest.mark.unit
def test_delete_file_removes_document(upload_dir: Path):
    document_dir = upload_dir / "document-01"
    document_dir.mkdir()
    (document_dir / "source.pdf").write_bytes(b"content")

    assert local.delete_file("document-01") is True
    assert not document_dir.exists()


@pytest.mark.unit
def test_delete_file_rejects_symlink_document_directory(upload_dir: Path):
    (upload_dir / "document-01").mkdir()

    with patch.object(Path, "is_symlink", return_value=True):
        with pytest.raises(local.FileStorageError, match="Invalid storage directory"):
            local.delete_file("document-01")
