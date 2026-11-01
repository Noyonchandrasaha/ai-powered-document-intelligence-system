import asyncio
import io
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TypedDict

from app.core.logging import get_logger
from app.features.document_extraction.data_processing.ocr.google_vision import (
    GoogleVisionOCR,
    OCRWord,
)
from PIL import (
    Image,
    ImageOps,
    ImageStat,
    UnidentifiedImageError,
)

logger = get_logger(__name__)

MAX_SOURCE_PIXELS = 40_000_000
MAX_OUTPUT_DIMENSION = 3000
MAX_OCR_BYTES = 7 * 1024 * 1024


class ImageProcessingError(Exception):
    """Raised when an image cannot be opened or decoded."""


class ImagePageResult(TypedDict):
    page_number: int
    total_pages: int
    status: str
    text: str
    confidence: float | None
    words: list[OCRWord]
    image_width: int | None
    image_height: int | None
    extraction_method: str
    error: str | None


def get_image_page_count(
    file_path: str | Path,
) -> int:
    """Return TIFF frame count; JPEG and PNG count as one page."""

    try:
        with Image.open(file_path) as image:
            if image.format not in {"JPEG", "PNG", "TIFF"}:
                raise ImageProcessingError("Unsupported image format.")

            return int(getattr(image, "n_frames", 1)) if image.format == "TIFF" else 1

    except (FileNotFoundError, ImageProcessingError):
        raise

    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as exc:
        raise ImageProcessingError("Unable to read image metadata.") from exc


def _encode_ocr_image(
    image: Image.Image,
) -> bytes:
    """Encode a page within the direct OCR request size budget."""

    # Prefer PNG to preserve sharp text and fine details.
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
        optimize=True,
    )

    if buffer.tell() <= MAX_OCR_BYTES:
        return buffer.getvalue()

    # Fall back to compressed JPEG.
    for _ in range(5):
        for quality in (90, 80, 65):
            buffer = io.BytesIO()

            image.save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=True,
            )

            if buffer.tell() <= MAX_OCR_BYTES:
                return buffer.getvalue()

        width, height = image.size

        if min(width, height) < 400:
            break

        image.thumbnail(
            (
                max(1, int(width * 0.85)),
                max(1, int(height * 0.85)),
            ),
            Image.Resampling.LANCZOS,
        )

    raise ImageProcessingError("Page exceeds OCR image size limit.")


def _prepare_frame(
    source: Image.Image,
    frame_index: int,
    enhance: bool,
) -> tuple[bytes, tuple[int, int]]:
    """Prepare one image frame for OCR."""

    source.seek(frame_index)

    width, height = source.size

    if width <= 0 or height <= 0 or width * height > MAX_SOURCE_PIXELS:
        raise ImageProcessingError("Image dimensions exceed safety limits.")

    # Correct EXIF orientation.
    oriented = ImageOps.exif_transpose(source)

    # Convert transparent images to RGB with white background.
    if "A" in oriented.getbands() or (oriented.mode == "P" and "transparency" in oriented.info):
        background = Image.new(
            "RGBA",
            oriented.size,
            "white",
        )

        background.alpha_composite(oriented.convert("RGBA"))

        processed = background.convert("RGB")

    else:
        processed = oriented.convert("RGB")

    # Limit resolution while preserving aspect ratio.
    processed.thumbnail(
        (
            MAX_OUTPUT_DIMENSION,
            MAX_OUTPUT_DIMENSION,
        ),
        Image.Resampling.LANCZOS,
    )

    # Apply contrast enhancement only when needed.
    if enhance:
        sample = processed.copy()

        sample.thumbnail(
            (256, 256),
            Image.Resampling.BILINEAR,
        )

        contrast_stddev = ImageStat.Stat(sample.convert("L")).stddev[0]

        if contrast_stddev < 40:
            processed = ImageOps.autocontrast(
                processed,
                cutoff=1,
            )

    image_bytes = _encode_ocr_image(processed)

    return image_bytes, processed.size


async def extract_image_pages(
    file_path: str | Path,
    ocr: GoogleVisionOCR,
    *,
    enhance: bool = True,
) -> AsyncIterator[ImagePageResult]:
    """
    Extract OCR text from JPG, PNG and multi-page TIFF.

    Yields one page result at a time.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        image = Image.open(path)

    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as exc:
        raise ImageProcessingError("Unable to open image.") from exc

    with image:
        if image.format not in {"JPEG", "PNG", "TIFF"}:
            raise ImageProcessingError("Unsupported image format.")

        total_pages = int(getattr(image, "n_frames", 1)) if image.format == "TIFF" else 1

        for frame_index in range(total_pages):
            page_number = frame_index + 1

            try:
                # Image processing is synchronous and CPU-bound.
                image_bytes, (width, height) = await asyncio.to_thread(
                    _prepare_frame,
                    image,
                    frame_index,
                    enhance,
                )

                # Call the existing Google Vision OCR client.
                result = await ocr.extract_image(image_bytes)

                yield {
                    "page_number": page_number,
                    "total_pages": total_pages,
                    "status": "completed",
                    "text": result["text"],
                    "confidence": result["confidence"],
                    "words": result["words"],
                    "image_width": width,
                    "image_height": height,
                    "extraction_method": "google_vision",
                    "error": None,
                }

            except Exception as exc:
                logger.exception(
                    "image_ocr_page_failed",
                    page_number=page_number,
                    error=str(exc),
                )

                yield {
                    "page_number": page_number,
                    "total_pages": total_pages,
                    "status": "failed",
                    "text": "",
                    "confidence": None,
                    "words": [],
                    "image_width": None,
                    "image_height": None,
                    "extraction_method": "google_vision",
                    "error": (f"Page processing failed: {type(exc).__name__}"),
                }
