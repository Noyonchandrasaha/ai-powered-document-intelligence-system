import asyncio
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, TypedDict

from app.core.logging import get_logger
from app.features.document_extraction.data_processing.csv import (
    extract_csv_chunks,
)
from app.features.document_extraction.data_processing.excel import (
    extract_excel_chunks,
)
from app.features.document_extraction.data_processing.image import (
    extract_image_pages,
)
from app.features.document_extraction.data_processing.ocr.google_vision import (
    GoogleVisionOCR,
    OCRWord,
)
from app.features.document_extraction.data_processing.pdf import (
    extract_pdf_pages,
    get_pdf_page_count,
)
from app.features.document_extraction.schemas import DocumentType

logger = get_logger(__name__)


async def _threaded(iterator: Iterator[Any]) -> AsyncIterator[Any]:
    """Advance CPU/blocking parsers off the event loop, one bounded unit at a time."""
    sentinel = object()
    # PyMuPDF's document/page objects are not thread-safe. Every iterator must
    # be opened, advanced, and closed by the same dedicated parser thread.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="document-parser") as executor:
        try:
            while True:
                item = await asyncio.get_running_loop().run_in_executor(
                    executor, next, iterator, sentinel
                )
                if item is sentinel:
                    return
                yield item
        finally:
            close = getattr(iterator, "close", None)
            if close:
                await asyncio.get_running_loop().run_in_executor(executor, close)


ResultStatus = Literal["completed", "partial", "failed"]

ResultKind = Literal[
    "page",
    "sheet_chunk",
    "csv_chunk",
]


class ProcessingPipelineError(Exception):
    """Raised when a processing job cannot be started."""


class ProcessingResult(TypedDict):
    document_id: str
    unit_id: str
    kind: ResultKind
    status: ResultStatus
    extraction_method: str

    page_number: int | None
    sheet_name: str | None
    sheet_index: int | None
    start_row: int | None
    end_row: int | None

    text: str
    native_text: str
    ocr_text: str

    text_blocks: list[dict[str, Any]]
    words: list[OCRWord]
    rows: list[dict[str, Any]]

    metadata: dict[str, Any]
    error: str | None


async def _process_pdf(
    document_id: str,
    file_path: Path,
    ocr: GoogleVisionOCR | None,
) -> AsyncIterator[ProcessingResult]:

    total_pages = await asyncio.to_thread(get_pdf_page_count, file_path)

    async for page in _threaded(extract_pdf_pages(file_path)):
        page_number = page["page_number"]
        native_text = page["text"]

        result: ProcessingResult = {
            "document_id": document_id,
            "unit_id": f"page:{page_number}",
            "kind": "page",
            "status": ("failed" if page["error"] else "completed"),
            "extraction_method": "pymupdf",
            "page_number": page_number,
            "sheet_name": None,
            "sheet_index": None,
            "start_row": None,
            "end_row": None,
            "text": native_text,
            "native_text": native_text,
            "ocr_text": "",
            "text_blocks": page["text_blocks"],
            "words": [],
            "rows": [],
            "metadata": {
                "total_pages": total_pages,
                "page_type": page["page_type"],
                "image_coverage": page["image_coverage"],
                "needs_ocr": page["needs_ocr"],
                "ocr_attempted": False,
                "native_coordinate_space": "pdf_points",
            },
            "error": page["error"],
        }

        if page["error"]:
            yield result
            await asyncio.sleep(0)
            continue

        if page["needs_ocr"]:
            if ocr is None:
                result["status"] = "partial" if native_text else "failed"
                result["error"] = "OCR required but Google Vision client is not configured."

            else:
                result["metadata"]["ocr_attempted"] = True

                try:
                    ocr_result = await ocr.extract_pdf_page(
                        file_path=file_path,
                        page_number=page_number,
                    )

                    result["ocr_text"] = ocr_result["text"]
                    result["words"] = ocr_result["words"]

                    result["metadata"]["ocr_confidence"] = ocr_result["confidence"]
                    result["metadata"]["ocr_image_width"] = ocr_result["image_width"]
                    result["metadata"]["ocr_image_height"] = ocr_result["image_height"]
                    result["metadata"]["ocr_coordinate_space"] = "rendered_image_pixels"

                    if native_text:
                        result["extraction_method"] = "pymupdf+google_vision"
                        result["metadata"]["requires_text_merge"] = bool(ocr_result["text"])
                    else:
                        result["text"] = ocr_result["text"]
                        result["extraction_method"] = "google_vision"

                except Exception as exc:
                    logger.exception(
                        "pipeline_pdf_ocr_failed",
                        document_id=document_id,
                        page_number=page_number,
                        error=str(exc),
                    )
                    result["status"] = "partial" if native_text else "failed"
                    result["error"] = f"Google Vision OCR failed: {type(exc).__name__}"

        yield result
        await asyncio.sleep(0)


async def _process_image(
    document_id: str,
    file_path: Path,
    ocr: GoogleVisionOCR | None,
) -> AsyncIterator[ProcessingResult]:

    if ocr is None:
        raise ProcessingPipelineError("Google Vision OCR is required for image processing.")

    async for page in extract_image_pages(
        file_path=file_path,
        ocr=ocr,
    ):
        yield {
            "document_id": document_id,
            "unit_id": f"page:{page['page_number']}",
            "kind": "page",
            "status": ("completed" if page["status"] == "completed" else "failed"),
            "extraction_method": "google_vision",
            "page_number": page["page_number"],
            "sheet_name": None,
            "sheet_index": None,
            "start_row": None,
            "end_row": None,
            "text": page["text"],
            "native_text": "",
            "ocr_text": page["text"],
            "text_blocks": [],
            "words": page["words"],
            "rows": [],
            "metadata": {
                "total_pages": page["total_pages"],
                "confidence": page["confidence"],
                "image_width": page["image_width"],
                "image_height": page["image_height"],
                "coordinate_space": "processed_image_pixels",
            },
            "error": page["error"],
        }


async def _process_excel(
    document_id: str,
    file_path: Path,
    chunk_size: int,
) -> AsyncIterator[ProcessingResult]:

    async for chunk in _threaded(
        extract_excel_chunks(
            file_path=file_path,
            chunk_size=chunk_size,
        )
    ):
        sheet_index = chunk["sheet_index"]
        start_row = chunk["start_row"]

        unit_id = f"sheet:{sheet_index}:row:{start_row if start_row is not None else 'empty'}"

        yield {
            "document_id": document_id,
            "unit_id": unit_id,
            "kind": "sheet_chunk",
            "status": ("completed" if chunk["status"] == "completed" else "failed"),
            "extraction_method": "excel_parser",
            "page_number": None,
            "sheet_name": chunk["sheet_name"],
            "sheet_index": sheet_index,
            "start_row": chunk["start_row"],
            "end_row": chunk["end_row"],
            "text": "",
            "native_text": "",
            "ocr_text": "",
            "text_blocks": [],
            "words": [],
            "rows": chunk["rows"],
            "metadata": {
                "row_count": len(chunk["rows"]),
                "file_format": file_path.suffix.lower(),
            },
            "error": chunk["error"],
        }

        await asyncio.sleep(0)


async def _process_csv(
    document_id: str,
    file_path: Path,
    chunk_size: int,
) -> AsyncIterator[ProcessingResult]:

    chunk_index = 0
    async for chunk in _threaded(
        extract_csv_chunks(
            file_path=file_path,
            chunk_size=chunk_size,
        )
    ):
        chunk_index += 1
        yield {
            "document_id": document_id,
            "unit_id": f"csv:chunk:{chunk_index}",
            "kind": "csv_chunk",
            "status": ("completed" if chunk["status"] == "completed" else "failed"),
            "extraction_method": "csv_parser",
            "page_number": None,
            "sheet_name": None,
            "sheet_index": None,
            "start_row": chunk["start_row"],
            "end_row": chunk["end_row"],
            "text": "",
            "native_text": "",
            "ocr_text": "",
            "text_blocks": [],
            "words": [],
            "rows": chunk["rows"],
            "metadata": {
                "encoding": chunk["encoding"],
                "delimiter": chunk["delimiter"],
                "row_count": len(chunk["rows"]),
            },
            "error": chunk["error"],
        }

        await asyncio.sleep(0)


async def process_document(
    document_id: str,
    file_path: str | Path,
    file_type: DocumentType,
    ocr: GoogleVisionOCR | None = None,
    *,
    excel_chunk_size: int = 200,
    csv_chunk_size: int = 200,
) -> AsyncIterator[ProcessingResult]:
    """
    Route a document to its appropriate processor.

    Yields normalized extraction results one unit at a time.

    This function does not save results to MongoDB.
    """

    path = Path(file_path)

    if not document_id.strip():
        raise ValueError("Document ID is required.")

    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    if excel_chunk_size < 1 or csv_chunk_size < 1:
        raise ValueError("Chunk sizes must be positive.")

    if file_type == DocumentType.PDF:
        async for result in _process_pdf(
            document_id,
            path,
            ocr,
        ):
            yield result

    elif file_type in {
        DocumentType.JPG,
        DocumentType.PNG,
        DocumentType.TIFF,
    }:
        async for result in _process_image(
            document_id,
            path,
            ocr,
        ):
            yield result

    elif file_type in {
        DocumentType.XLSX,
        DocumentType.XLS,
    }:
        async for result in _process_excel(
            document_id,
            path,
            excel_chunk_size,
        ):
            yield result

    elif file_type == DocumentType.CSV:
        async for result in _process_csv(
            document_id,
            path,
            csv_chunk_size,
        ):
            yield result

    else:
        raise ProcessingPipelineError(f"Unsupported document type: {file_type}")
