import asyncio
import base64
import io
from pathlib import Path
from types import TracebackType
from typing import Any, TypedDict, cast

import httpx
import pymupdf
from app.core.config import get_settings
from app.core.logging import get_logger
from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from PIL import Image

logger = get_logger(__name__)

VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"

# Keep base64 JSON requests below Google's 10 MB limit.
MAX_IMAGE_BYTES = 7 * 1024 * 1024
GOOGLE_VISION_SCOPE = "https://www.googleapis.com/auth/cloud-vision"

RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


class VisionOCRError(Exception):
    """Raised when Google Vision OCR processing fails."""


class OCRWord(TypedDict):
    text: str
    confidence: float | None
    bounding_box: list[dict[str, int]]


class OCRResult(TypedDict):
    text: str
    confidence: float | None
    words: list[OCRWord]
    image_width: int | None
    image_height: int | None
    extraction_method: str


def render_pdf_page(
    file_path: str | Path,
    page_number: int,
    dpi: int = 200,
) -> bytes:
    """
    Render a single PDF page to compressed JPEG bytes.

    Page numbering starts at 1.
    """

    if dpi < 72 or dpi > 300:
        raise ValueError("DPI must be between 72 and 300.")

    try:
        with pymupdf.open(str(file_path)) as document:  # type: ignore[no-untyped-call]
            if document.needs_pass:
                raise VisionOCRError("Password-protected PDF is not supported.")

            if not 1 <= page_number <= document.page_count:
                raise ValueError("Invalid PDF page number.")

            page = document.load_page(page_number - 1)

            # Restrict rendering resolution for huge pages.
            width = page.rect.width * dpi / 72
            height = page.rect.height * dpi / 72

            if width <= 0 or height <= 0:
                raise VisionOCRError("PDF page has invalid dimensions.")

            max_pixels = 12_000_000
            scale = min(
                1.0,
                3000 / max(width, height),
                (max_pixels / (width * height)) ** 0.5,
            )

            matrix = pymupdf.Matrix(  # type: ignore[no-untyped-call]
                dpi * scale / 72,
                dpi * scale / 72,
            )

            pixmap = page.get_pixmap(
                matrix=matrix,
                colorspace=pymupdf.csRGB,
                alpha=False,
            )

            image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )

            for quality in (85, 75, 60):
                buffer = io.BytesIO()

                image.save(
                    buffer,
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                )

                image_bytes = buffer.getvalue()

                if len(image_bytes) <= MAX_IMAGE_BYTES:
                    return image_bytes

            raise VisionOCRError("Rendered PDF page exceeds OCR size limit.")

    except (ValueError, VisionOCRError):
        raise

    except Exception as exc:
        raise VisionOCRError(f"Failed to render PDF page {page_number}.") from exc


def parse_vision_response(
    response_data: dict[str, Any],
) -> OCRResult:
    """
    Convert Google Vision response into structured OCR data.
    """

    responses = response_data.get("responses", [])

    if not responses:
        raise VisionOCRError("Google Vision returned an empty response.")

    result = responses[0]

    if result.get("error"):
        error = result["error"]
        raise VisionOCRError(error.get("message", "Google Vision OCR failed."))

    annotation = result.get("fullTextAnnotation", {})

    text = annotation.get("text", "")
    pages = annotation.get("pages", [])

    words: list[OCRWord] = []
    page_confidences: list[float] = []

    for page in pages:
        if page.get("confidence") is not None:
            page_confidences.append(float(page["confidence"]))

        for block in page.get("blocks", []):
            for paragraph in block.get("paragraphs", []):
                for word in paragraph.get("words", []):
                    word_text = "".join(
                        symbol.get("text", "") for symbol in word.get("symbols", [])
                    )

                    if not word_text:
                        continue

                    vertices = word.get("boundingBox", {}).get("vertices", [])

                    bounding_box = [
                        {
                            "x": int(vertex.get("x", 0)),
                            "y": int(vertex.get("y", 0)),
                        }
                        for vertex in vertices
                    ]

                    confidence = word.get("confidence")

                    words.append(
                        {
                            "text": word_text,
                            "confidence": (float(confidence) if confidence is not None else None),
                            "bounding_box": bounding_box,
                        }
                    )

    confidence = sum(page_confidences) / len(page_confidences) if page_confidences else None

    first_page = pages[0] if pages else {}

    return {
        "text": text,
        "confidence": confidence,
        "words": words,
        "image_width": first_page.get("width"),
        "image_height": first_page.get("height"),
        "extraction_method": "google_vision",
    }


class GoogleVisionOCR:
    """
    Async Google Vision OCR client.

    Reuse one instance for multiple images/pages.
    """

    def __init__(
        self,
        credentials_file: str | Path | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        credential_path = Path(
            credentials_file or get_settings().google_vision_credentials_file
        ).expanduser()

        if not credential_path.is_file():
            raise FileNotFoundError(f"Google Vision credential file not found: {credential_path}")

        try:
            self.credentials = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
                str(credential_path),
                scopes=[GOOGLE_VISION_SCOPE],
            )
        except (GoogleAuthError, OSError, ValueError) as exc:
            raise ValueError("Invalid Google Vision service-account credentials.") from exc

        self.max_retries = max_retries

        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

    async def __aenter__(self) -> "GoogleVisionOCR":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.client.aclose()

    async def _get_access_token(self) -> str:
        if not self.credentials.valid:
            try:
                await asyncio.to_thread(
                    self.credentials.refresh,
                    Request(),
                )
            except GoogleAuthError as exc:
                raise VisionOCRError("Google Vision authentication failed.") from exc

        if not self.credentials.token:
            raise VisionOCRError("Google Vision authentication token is unavailable.")

        return cast(str, self.credentials.token)

    async def extract_image(
        self,
        image_bytes: bytes,
    ) -> OCRResult:
        """
        Extract text from JPEG/PNG image bytes.
        """

        if not image_bytes:
            raise VisionOCRError("Image content cannot be empty.")

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise VisionOCRError("Image exceeds direct OCR request size limit.")

        encoded_image = base64.b64encode(image_bytes).decode("ascii")

        payload = {
            "requests": [
                {
                    "image": {
                        "content": encoded_image,
                    },
                    "features": [
                        {
                            "type": "DOCUMENT_TEXT_DETECTION",
                        }
                    ],
                }
            ]
        }

        for attempt in range(self.max_retries + 1):
            try:
                access_token = await self._get_access_token()
                response = await self.client.post(
                    VISION_API_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                    },
                )

            except httpx.RequestError as exc:
                if attempt < self.max_retries:
                    await asyncio.sleep(2**attempt)
                    continue

                raise VisionOCRError("Google Vision request failed.") from exc

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                await asyncio.sleep(2**attempt)
                continue

            if response.is_error:
                raise VisionOCRError(f"Google Vision returned HTTP {response.status_code}.")

            try:
                response_data = response.json()

            except ValueError as exc:
                raise VisionOCRError("Invalid Google Vision JSON response.") from exc

            return parse_vision_response(response_data)

        raise VisionOCRError("Google Vision OCR retries exhausted.")

    async def extract_pdf_page(
        self,
        file_path: str | Path,
        page_number: int,
        dpi: int = 200,
    ) -> OCRResult:
        """
        Render one PDF page locally and perform OCR.
        """

        image_bytes = await asyncio.to_thread(
            render_pdf_page,
            file_path,
            page_number,
            dpi,
        )

        result = await self.extract_image(image_bytes)

        logger.info(
            "google_vision_ocr_completed",
            page_number=page_number,
        )

        return result
