import asyncio
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.features.document_extraction.data_processing.image import (
    MAX_SOURCE_PIXELS,
    ImageProcessingError,
    _prepare_frame,
    extract_image_pages,
    get_image_page_count,
)
from PIL import Image


def create_image(
    file_path: Path,
    image_format: str,
    size: tuple[int, int] = (20, 20),
    frame_count: int = 1,
) -> None:
    frames = [Image.new("RGB", size, color="white") for _ in range(frame_count)]
    frames[0].save(
        file_path,
        format=image_format,
        save_all=frame_count > 1,
        append_images=frames[1:],
    )


def ocr_result(text: str = "Recognized text") -> dict:
    return {
        "text": text,
        "confidence": 0.95,
        "words": [],
        "image_width": 20,
        "image_height": 20,
        "extraction_method": "google_vision",
    }


@pytest.mark.unit
@pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
def test_get_image_page_count_returns_one_for_single_frame_images(
    workspace_tmp_path: Path,
    image_format: str,
):
    file_path = workspace_tmp_path / f"image.{image_format.lower()}"
    create_image(file_path, image_format)

    assert get_image_page_count(file_path) == 1


@pytest.mark.unit
def test_get_image_page_count_returns_tiff_frame_count(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "image.tiff"
    create_image(file_path, "TIFF", frame_count=2)

    assert get_image_page_count(file_path) == 2


@pytest.mark.unit
def test_get_image_page_count_rejects_missing_file(workspace_tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        get_image_page_count(workspace_tmp_path / "missing.png")


@pytest.mark.unit
def test_get_image_page_count_rejects_corrupted_file(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "corrupted.png"
    file_path.write_bytes(b"not an image")

    with pytest.raises(ImageProcessingError, match="Unable to read image"):
        get_image_page_count(file_path)


@pytest.mark.unit
def test_get_image_page_count_rejects_unsupported_format(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "image.gif"
    create_image(file_path, "GIF")

    with pytest.raises(ImageProcessingError, match="Unsupported image format"):
        get_image_page_count(file_path)


@pytest.mark.unit
def test_prepare_frame_converts_transparent_image_to_rgb(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "transparent.png"
    image = Image.new("RGBA", (20, 20), color=(255, 0, 0, 128))
    image.save(file_path, format="PNG")

    with Image.open(file_path) as source:
        image_bytes, size = _prepare_frame(source, 0, enhance=False)

    assert size == (20, 20)
    with Image.open(io.BytesIO(image_bytes)) as processed:
        assert processed.mode == "RGB"


@pytest.mark.unit
def test_prepare_frame_rejects_oversized_source_image(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "image.png"
    create_image(file_path, "PNG", size=(20, 20))

    with Image.open(file_path) as source:
        with patch(
            "app.features.document_extraction.data_processing.image.MAX_SOURCE_PIXELS",
            10,
        ):
            with pytest.raises(ImageProcessingError, match="dimensions exceed"):
                _prepare_frame(source, 0, enhance=False)

    assert MAX_SOURCE_PIXELS > 0


@pytest.mark.unit
def test_extract_image_pages_returns_completed_result(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "image.png"
    create_image(file_path, "PNG")
    ocr = MagicMock()
    ocr.extract_image = AsyncMock(return_value=ocr_result())

    results = asyncio.run(collect_results(extract_image_pages(file_path, ocr, enhance=False)))

    assert results == [
        {
            "page_number": 1,
            "total_pages": 1,
            "status": "completed",
            "text": "Recognized text",
            "confidence": 0.95,
            "words": [],
            "image_width": 20,
            "image_height": 20,
            "extraction_method": "google_vision",
            "error": None,
        }
    ]
    ocr.extract_image.assert_awaited_once()


@pytest.mark.unit
def test_extract_image_pages_processes_all_tiff_frames(workspace_tmp_path: Path):
    file_path = workspace_tmp_path / "image.tiff"
    create_image(file_path, "TIFF", frame_count=2)
    ocr = MagicMock()
    ocr.extract_image = AsyncMock(side_effect=[ocr_result("one"), ocr_result("two")])

    results = asyncio.run(collect_results(extract_image_pages(file_path, ocr)))

    assert [result["page_number"] for result in results] == [1, 2]
    assert [result["text"] for result in results] == ["one", "two"]
    assert all(result["total_pages"] == 2 for result in results)
    assert ocr.extract_image.await_count == 2


@pytest.mark.unit
def test_extract_image_pages_yields_failed_result_and_logs_error(
    workspace_tmp_path: Path,
):
    file_path = workspace_tmp_path / "image.png"
    create_image(file_path, "PNG")
    ocr = MagicMock()
    ocr.extract_image = AsyncMock(side_effect=RuntimeError("OCR unavailable"))

    with patch(
        "app.features.document_extraction.data_processing.image.logger.exception"
    ) as log_exception:
        results = asyncio.run(collect_results(extract_image_pages(file_path, ocr)))

    assert results[0]["status"] == "failed"
    assert results[0]["text"] == ""
    assert results[0]["error"] == "Page processing failed: RuntimeError"
    log_exception.assert_called_once_with(
        "image_ocr_page_failed",
        page_number=1,
        error="OCR unavailable",
    )


@pytest.mark.unit
def test_extract_image_pages_rejects_missing_file(
    workspace_tmp_path: Path,
):
    with pytest.raises(FileNotFoundError, match="Image not found"):
        asyncio.run(
            collect_results(
                extract_image_pages(
                    workspace_tmp_path / "missing.png",
                    MagicMock(),
                )
            )
        )


async def collect_results(generator) -> list[dict]:
    return [result async for result in generator]
