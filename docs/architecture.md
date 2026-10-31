# Backend Architecture

## Request flow

1. Upload validates extension, signature, integrity, and size.
2. The source file is stored under an isolated document directory.
3. MongoDB stores document metadata with `pending` status.
4. FastAPI schedules document processing after the upload response.
5. The processing service runs PDF/OCR/spreadsheet extraction and persists
   normalized units in `extraction_results`.
6. The document status and `processing_history` are updated.
7. Invoice extraction builds bounded source context, calls OpenAI, validates
   evidence and financial consistency, checks duplicate invoice numbers, and
   persists the result.
8. When semantic search is enabled, invoice data is embedded with OpenAI and
   stored alongside the extraction for vector similarity search.

## Collections

- `documents`: upload metadata, storage path, status, page count, and errors.
- `extraction_results`: page/sheet/CSV units with text, OCR words, rows, and
  source metadata.
- `processing_history`: every processing attempt and its outcome.
- `invoice_extractions`: validated invoice data, evidence, review flags, and
  model/prompt metadata, plus optional embeddings.

## Reliability boundaries

The current worker adapter is FastAPI `BackgroundTasks`, with job state stored
in MongoDB. For multi-process or multi-host production deployment, replace the
adapter with a durable queue such as Celery/RQ/Redis while keeping
`DocumentProcessingService.process_document()` as the job contract.
