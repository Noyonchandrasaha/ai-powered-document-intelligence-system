
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict

import pymupdf
from app.core.logging import get_logger

logger = get_logger(__name__)


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be processed."""


class PDFPageResult(TypedDict):
    page_number: int
    page_type: str
    extraction_method: str
    text: str
    text_blocks: list[dict[str, Any]]
    image_coverage: float
    needs_ocr: bool
    error: str | None


def get_pdf_page_count(
    file_path: str | Path,
) -> int:
    try:
        with pymupdf.open(str(file_path)) as document:  # type: ignore[no-untyped-call]
            if document.needs_pass:
                raise PDFProcessingError(
                    "Password-protected PDF is not supported."
                )

            return int(document.page_count)

    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(
            "Unable to read PDF page count."
        ) from exc


def calculate_image_coverage(
    page: pymupdf.Page,
) -> float:
    page_rect = page.cropbox
    page_area = page_rect.get_area()

    if page_area <= 0:
        return 0.0

    covered_area = 0.0

    for image in page.get_image_info(
        hashes=False,
        xrefs=False,
    ):
        image_rect = pymupdf.Rect(image["bbox"])  # type: ignore[no-untyped-call]
        visible_rect = image_rect & page_rect

        if not visible_rect.is_empty:
            covered_area += float(visible_rect.get_area())

    # Approximation: overlapping images may be counted twice.
    return float(
        round(
            min(covered_area / page_area, 1.0),
            3,
        )
    )


def classify_pdf_page(
    page: pymupdf.Page,
    text: str,
    image_coverage: float,
) -> tuple[str, bool]:

    has_text = bool(text.strip())
    has_images = image_coverage > 0

    if has_text and image_coverage >= 0.25:
        return "mixed", True

    if has_text:
        return "digital", False

    if has_images:
        return "scanned", True

    # Vector graphics may contain outlined text.
    if page.get_drawings():
        return "unknown", True

    return "blank", False


def extract_pdf_pages(
    file_path: str | Path,
    start_page: int = 1,
    end_page: int | None = None,
) -> Iterator[PDFPageResult]:

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {path}"
        )

    try:
        document = pymupdf.open(str(path))  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise PDFProcessingError(
            "Unable to open PDF document."
        ) from exc

    with document:

        if document.needs_pass:
            raise PDFProcessingError(
                "Password-protected PDF is not supported."
            )

        total_pages = document.page_count
        last_page = (
            total_pages if end_page is None else end_page
        )

        if not (
            1 <= start_page <= last_page <= total_pages
        ):
            raise ValueError(
                "Invalid PDF page range."
            )

        for page_index in range(
            start_page - 1,
            last_page,
        ):
            page_number = page_index + 1

            try:
                page = document.load_page(page_index)  # type: ignore[no-untyped-call]

                raw_blocks = page.get_text(
                    "blocks",
                    sort=True,
                )

                text_blocks = []

                for block in raw_blocks:
                    # Block type 0 represents text.
                    if block[6] != 0:
                        continue

                    content = block[4].strip()

                    if not content:
                        continue

                    text_blocks.append({
                        "text": content,
                        "bbox": [
                            round(float(value), 2)
                            for value in block[:4]
                        ],
                    })

                extracted_text = "\n".join(
                    block["text"]
                    for block in text_blocks
                )

                image_coverage = calculate_image_coverage(
                    page
                )

                page_type, needs_ocr = classify_pdf_page(
                    page,
                    extracted_text,
                    image_coverage,
                )

                yield {
                    "page_number": page_number,
                    "page_type": page_type,
                    "extraction_method": "pymupdf",
                    "text": extracted_text,
                    "text_blocks": text_blocks,
                    "image_coverage": image_coverage,
                    "needs_ocr": needs_ocr,
                    "error": None,
                }

            except Exception as exc:
                logger.exception(
                    "pdf_page_processing_failed",
                    page_number=page_number,
                    error=str(exc),
                )

                yield {
                    "page_number": page_number,
                    "page_type": "error",
                    "extraction_method": "pymupdf",
                    "text": "",
                    "text_blocks": [],
                    "image_coverage": 0.0,
                    "needs_ocr": False,
                    "error": (
                        "Page extraction failed: "
                        f"{type(exc).__name__}"
                    ),
                }
