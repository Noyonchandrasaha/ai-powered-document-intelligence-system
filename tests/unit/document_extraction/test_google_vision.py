import asyncio
import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from app.features.document_extraction.data_processing.ocr.google_vision import (
    MAX_IMAGE_BYTES,
    GoogleVisionOCR,
    VisionOCRError,
    parse_vision_response,
)


def vision_response() -> dict:
    return {
        "responses": [
            {
                "fullTextAnnotation": {
                    "text": "Invoice 123",
                    "pages": [
                        {
                            "width": 800,
                            "height": 600,
                            "confidence": 0.8,
                            "blocks": [
                                {
                                    "paragraphs": [
                                        {
                                            "words": [
                                                {
                                                    "confidence": 0.9,
                                                    "symbols": [
                                                        {"text": "Invoice"},
                                                    ],
                                                    "boundingBox": {
                                                        "vertices": [
                                                            {"x": 1, "y": 2},
                                                        ]
                                                    },
                                                },
                                                {
                                                    "symbols": [{"text": "123"}],
                                                },
                                            ]
                                        }
                                    ]
                                }
                            ],
                        }
                    ],
                }
            }
        ]
    }


@pytest.fixture
def ocr_client(workspace_tmp_path: Path):
    credentials_file = workspace_tmp_path / "Vision-api.text"
    credentials_file.write_text("{}", encoding="utf-8")

    with patch(
        "app.features.document_extraction.data_processing.ocr.google_vision.service_account.Credentials.from_service_account_file"
    ) as load_credentials:
        load_credentials.return_value = MagicMock()
        client = GoogleVisionOCR(credentials_file=credentials_file)

    yield client
    asyncio.run(client.client.aclose())


@pytest.mark.unit
def test_parse_vision_response_returns_text_words_and_confidence():
    result = parse_vision_response(vision_response())

    assert result["text"] == "Invoice 123"
    assert result["confidence"] == 0.8
    assert result["image_width"] == 800
    assert result["image_height"] == 600
    assert result["extraction_method"] == "google_vision"
    assert result["words"][0] == {
        "text": "Invoice",
        "confidence": 0.9,
        "bounding_box": [{"x": 1, "y": 2}],
    }
    assert result["words"][1]["confidence"] is None


@pytest.mark.unit
def test_parse_vision_response_rejects_empty_response():
    with pytest.raises(VisionOCRError, match="empty response"):
        parse_vision_response({})


@pytest.mark.unit
def test_parse_vision_response_raises_api_error():
    with pytest.raises(VisionOCRError, match="Permission denied"):
        parse_vision_response({"responses": [{"error": {"message": "Permission denied"}}]})


@pytest.mark.unit
def test_google_vision_requires_existing_credentials_file(workspace_tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="credential file not found"):
        GoogleVisionOCR(credentials_file=workspace_tmp_path / "missing.text")


@pytest.mark.unit
def test_google_vision_rejects_negative_retries(ocr_client: GoogleVisionOCR):
    with pytest.raises(ValueError, match="cannot be negative"):
        GoogleVisionOCR(credentials_file="unused.text", max_retries=-1)


@pytest.mark.unit
def test_extract_image_rejects_empty_content(ocr_client: GoogleVisionOCR):
    with pytest.raises(VisionOCRError, match="cannot be empty"):
        asyncio.run(ocr_client.extract_image(b""))


@pytest.mark.unit
def test_extract_image_rejects_oversized_content(ocr_client: GoogleVisionOCR):
    with pytest.raises(VisionOCRError, match="size limit"):
        asyncio.run(ocr_client.extract_image(b"x" * (MAX_IMAGE_BYTES + 1)))


@pytest.mark.unit
def test_get_access_token_refreshes_service_account_credentials(
    ocr_client: GoogleVisionOCR,
):
    ocr_client.credentials.valid = False
    ocr_client.credentials.token = None

    def refresh_credentials(_request):
        ocr_client.credentials.token = "refreshed-token"

    ocr_client.credentials.refresh = MagicMock(side_effect=refresh_credentials)

    token = asyncio.run(ocr_client._get_access_token())

    assert token == "refreshed-token"
    ocr_client.credentials.refresh.assert_called_once()


@pytest.mark.unit
def test_get_access_token_rejects_missing_token(ocr_client: GoogleVisionOCR):
    ocr_client.credentials.valid = True
    ocr_client.credentials.token = None

    with pytest.raises(VisionOCRError, match="token is unavailable"):
        asyncio.run(ocr_client._get_access_token())


@pytest.mark.unit
def test_extract_image_sends_authenticated_request(ocr_client: GoogleVisionOCR):
    ocr_client._get_access_token = AsyncMock(return_value="access-token")
    ocr_client.client.post = AsyncMock(
        return_value=httpx.Response(200, json=vision_response())
    )

    result = asyncio.run(ocr_client.extract_image(b"image-bytes"))

    request = ocr_client.client.post.call_args
    payload = request.kwargs["json"]
    assert request.args[0] == "https://vision.googleapis.com/v1/images:annotate"
    assert request.kwargs["headers"] == {"Authorization": "Bearer access-token"}
    assert payload["requests"][0]["features"][0]["type"] == "DOCUMENT_TEXT_DETECTION"
    assert payload["requests"][0]["image"]["content"] == base64.b64encode(
        b"image-bytes"
    ).decode("ascii")
    assert result["text"] == "Invoice 123"


@pytest.mark.unit
def test_extract_image_retries_retryable_response(ocr_client: GoogleVisionOCR):
    ocr_client._get_access_token = AsyncMock(return_value="access-token")
    ocr_client.client.post = AsyncMock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=vision_response()),
        ]
    )

    with patch(
        "app.features.document_extraction.data_processing.ocr.google_vision.asyncio.sleep",
        new_callable=AsyncMock,
    ) as sleep:
        result = asyncio.run(ocr_client.extract_image(b"image-bytes"))

    assert result["text"] == "Invoice 123"
    assert ocr_client.client.post.await_count == 2
    sleep.assert_awaited_once_with(1)


@pytest.mark.unit
def test_extract_image_rejects_non_retryable_http_error(ocr_client: GoogleVisionOCR):
    ocr_client._get_access_token = AsyncMock(return_value="access-token")
    ocr_client.client.post = AsyncMock(return_value=httpx.Response(400))

    with pytest.raises(VisionOCRError, match="HTTP 400"):
        asyncio.run(ocr_client.extract_image(b"image-bytes"))


@pytest.mark.unit
def test_extract_image_rejects_invalid_json_response(ocr_client: GoogleVisionOCR):
    ocr_client._get_access_token = AsyncMock(return_value="access-token")
    ocr_client.client.post = AsyncMock(
        return_value=httpx.Response(200, content=b"not-json")
    )

    with pytest.raises(VisionOCRError, match="Invalid Google Vision JSON"):
        asyncio.run(ocr_client.extract_image(b"image-bytes"))


@pytest.mark.unit
def test_extract_pdf_page_renders_and_logs_completion(ocr_client: GoogleVisionOCR):
    ocr_client.extract_image = AsyncMock(return_value=parse_vision_response(vision_response()))

    with (
        patch(
            "app.features.document_extraction.data_processing.ocr.google_vision.render_pdf_page",
            return_value=b"rendered-image",
        ) as render,
        patch(
            "app.features.document_extraction.data_processing.ocr.google_vision.logger.info"
        ) as log_info,
    ):
        result = asyncio.run(ocr_client.extract_pdf_page("document.pdf", 2))

    render.assert_called_once_with("document.pdf", 2, 200)
    ocr_client.extract_image.assert_awaited_once_with(b"rendered-image")
    log_info.assert_called_once_with(
        "google_vision_ocr_completed",
        page_number=2,
    )
    assert result["text"] == "Invoice 123"
