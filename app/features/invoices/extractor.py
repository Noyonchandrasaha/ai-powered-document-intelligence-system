from app.core.logging import get_logger
from app.features.invoices.context import InvoiceContext
from app.features.invoices.prompts import (
    INVOICE_EXTRACTION_SYSTEM_PROMPT,
    INVOICE_PROMPT_VERSION,
    build_invoice_user_prompt,
)
from app.features.invoices.schemas import InvoiceAIResult
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import ValidationError

logger = get_logger(__name__)


class InvoiceExtractionError(Exception):
    """Base exception for invoice AI extraction."""


class InvoiceExtractionTemporaryError(InvoiceExtractionError):
    """Retryable AI service failure."""


class InvoiceExtractionRefusalError(InvoiceExtractionError):
    """Raised when the AI model refuses extraction."""


class InvoiceExtractionResponseError(InvoiceExtractionError):
    """Raised when the model response is invalid or incomplete."""


class OpenAIInvoiceExtractor:
    """
    Extract structured invoice information using OpenAI.

    Responsibilities:
    - Send document context to OpenAI
    - Enforce the InvoiceAIResult schema
    - Handle API errors and refusals
    - Return parsed invoice information

    Does not:
    - Access MongoDB
    - Perform financial calculations
    - Verify source evidence
    - Persist extraction results
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small",
        max_output_tokens: int = 8192,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
    ) -> None:

        if not model.strip():
            raise ValueError("OpenAI model is required.")

        if not embedding_model.strip():
            raise ValueError("OpenAI embedding model is required.")

        if max_output_tokens < 256:
            raise ValueError("max_output_tokens must be at least 256.")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        self.client = client
        self.model = model
        self.embedding_model = embedding_model
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @staticmethod
    def _has_refusal(response: object) -> bool:
        """Check whether the response contains a refusal."""

        for output in getattr(response, "output", []):
            if getattr(output, "type", None) != "message":
                continue

            for content in getattr(output, "content", []):
                if getattr(content, "type", None) == "refusal":
                    return True

        return False

    async def extract(
        self,
        context: InvoiceContext,
    ) -> InvoiceAIResult:
        """
        Extract invoice information from a prepared context.

        The result is schema-validated but not yet
        verified against original document evidence.
        """

        user_prompt = build_invoice_user_prompt(context)

        logger.info(
            "invoice_extraction_started",
            document_id=context.document_id,
            model=self.model,
            prompt_version=INVOICE_PROMPT_VERSION,
        )

        try:
            response = await self.client.with_options(
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
            ).responses.parse(
                model=self.model,
                instructions=INVOICE_EXTRACTION_SYSTEM_PROMPT,
                input=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                text_format=InvoiceAIResult,
                max_output_tokens=self.max_output_tokens,
                store=False,
            )

        except RateLimitError as exc:
            logger.warning(
                "openai_rate_limit_reached",
                document_id=context.document_id,
            )

            raise InvoiceExtractionTemporaryError("OpenAI rate limit exceeded.") from exc

        except (APITimeoutError, APIConnectionError) as exc:
            logger.warning(
                "openai_connection_failure",
                document_id=context.document_id,
            )

            raise InvoiceExtractionTemporaryError("OpenAI connection failed or timed out.") from exc

        except APIStatusError as exc:
            logger.error(
                "openai_api_error",
                document_id=context.document_id,
                status_code=exc.status_code,
                request_id=exc.request_id,
            )

            if exc.status_code in {408, 409} or (exc.status_code >= 500):
                raise InvoiceExtractionTemporaryError("Temporary OpenAI service failure.") from exc

            raise InvoiceExtractionError(
                f"OpenAI request rejected (HTTP {exc.status_code})."
            ) from exc

        except APIError as exc:
            logger.exception(
                "openai_api_failure",
                document_id=context.document_id,
            )

            raise InvoiceExtractionError("OpenAI extraction request failed.") from exc

        except (ValidationError, ValueError) as exc:
            logger.warning(
                "openai_response_validation_failed",
                document_id=context.document_id,
            )

            raise InvoiceExtractionResponseError(
                "OpenAI returned an invalid or unparseable structured response."
            ) from exc

        if response.status != "completed":
            logger.warning(
                "openai_response_incomplete",
                document_id=context.document_id,
                response_id=response.id,
                response_status=response.status,
            )

            raise InvoiceExtractionResponseError("OpenAI did not complete invoice extraction.")

        if self._has_refusal(response):
            raise InvoiceExtractionRefusalError("OpenAI refused the extraction request.")

        parsed_result = response.output_parsed

        if parsed_result is None:
            raise InvoiceExtractionResponseError("OpenAI returned no parsed invoice result.")

        logger.info(
            "invoice_extraction_completed",
            document_id=context.document_id,
            response_id=response.id,
            is_invoice=parsed_result.is_invoice,
        )

        return parsed_result

    async def embed(self, text: str) -> list[float]:
        """Create a vector used by semantic invoice search."""

        if not text.strip():
            raise ValueError("Embedding text is required.")

        try:
            response = await self.client.with_options(
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
            ).embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float",
            )
        except RateLimitError as exc:
            raise InvoiceExtractionTemporaryError(
                "OpenAI rate limit exceeded while creating an embedding."
            ) from exc
        except (APITimeoutError, APIConnectionError) as exc:
            raise InvoiceExtractionTemporaryError(
                "OpenAI embedding request failed or timed out."
            ) from exc
        except APIError as exc:
            raise InvoiceExtractionError("OpenAI embedding request failed.") from exc

        if not response.data or not response.data[0].embedding:
            raise InvoiceExtractionResponseError("OpenAI returned an empty embedding.")

        return [float(value) for value in response.data[0].embedding]
