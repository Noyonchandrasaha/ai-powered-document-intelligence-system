# AI-Powered Document Intelligence Backend

FastAPI backend for processing business documents, extracting invoice data with
OpenAI, validating the result against source evidence, and storing searchable
results in MongoDB. The frontend is intentionally out of scope for this
assignment implementation.

## Architecture

```text
Upload API
   |
   v
Local file storage + Mongo document metadata
   |
   v
FastAPI BackgroundTasks
   |
   +--> PDF/PyMuPDF, Google Vision OCR, XLS/XLSX, CSV pipeline
   |        |
   |        +--> extraction_results collection
   |        +--> processing_history collection
   |        +--> document status update
   |
   v
Invoice context -> OpenAI Responses API -> source/evidence validation
   |
   +--> consistency and duplicate checks
   +--> invoice_extractions collection
```

The backend uses MongoDB for metadata, normalized extraction units, processing
history, and validated invoice results. `BackgroundTasks` keeps OCR and AI work
outside the upload request; a durable queue such as Celery/Redis can replace
this adapter without changing the processing pipeline contract.

## Local setup

1. Create and activate a Python 3.12 virtual environment.
2. Install dependencies:

   ```powershell
   python -m pip install -e ".[dev]"
   ```

3. Copy `.env.example` to `.env` and configure MongoDB, OpenAI, and Google
   Vision credentials.
4. Run MongoDB, then start the API:

   ```powershell
   uvicorn app.main:app --reload
   ```

Or run MongoDB and the API together:

```powershell
docker compose up --build
```

Never commit `.env`, service-account credentials, or API keys.

## API endpoints

### Documents

- `POST /api/v1/documents/upload` — upload PDF, JPG, PNG, TIFF, XLSX, XLS, or CSV files.
- `GET /api/v1/documents/{document_id}` — document metadata.
- `GET /api/v1/documents/{document_id}/status` — processing status and errors.
- `GET /api/v1/documents/{document_id}/history` — processing attempts.
- `POST /api/v1/documents/{document_id}/reprocess` — queue failed/partial processing again.
- `GET /api/v1/documents/search` — keyword/status filtering.
- `DELETE /api/v1/documents/{document_id}` — delete metadata and local source file.

### Invoice extraction

- `POST /api/v1/invoices/documents/{document_id}/extract` — run structured invoice extraction.
- `GET /api/v1/invoices/extractions/{extraction_id}` — retrieve one result.
- `GET /api/v1/invoices/documents/{document_id}/latest` — retrieve latest result.
- `GET /api/v1/invoices/search` — search invoice number, supplier, customer, or document ID.
- `GET /api/v1/invoices/search/semantic` — embedding-based invoice search.
- `PATCH /api/v1/invoices/extractions/{extraction_id}` — apply a manual correction.

Health endpoints:

- `GET /api/v1/health/live`
- `GET /api/v1/health/ready`
- `GET /`

## Processing and validation

The pipeline supports digital PDF text, scanned PDF OCR, image OCR, Excel
workbooks, legacy XLS, and CSV chunks. It preserves page, sheet, row, and cell
references for evidence validation. Invoice validation handles missing and
uncertain values, unsupported AI claims, numeric/date/currency evidence,
subtotal/tax/total consistency, line-item consistency, duplicate invoice
numbers, and manual-review status.

Set `APP_SEMANTIC_SEARCH_ENABLED=true` to create OpenAI embeddings during
invoice extraction and enable the semantic search endpoint. The portable local
implementation ranks stored vectors in Python; MongoDB Atlas Vector Search is
recommended when the collection grows large.

## Quality checks

```powershell
python -m pytest --cov=app --cov-report=term-missing --cov-fail-under=80
python -m ruff check .
python -m mypy app
```

Tests are organized under `tests/unit` and `tests/integration`. Integration
tests use controlled Mongo/OpenAI adapters and do not require live external
services.

## Production notes

- Set `APP_ENVIRONMENT=production` and restrict `APP_ALLOWED_HOSTS`.
- Keep `APP_OPENAI_API_KEY` and Google service-account credentials outside git.
- Mount `Vision-api.text` into `/app/secrets/Vision-api.text` when using Docker.
- Put the API behind TLS and an authenticated gateway.
- For high-volume deployments, run the processing service behind a durable
  queue/worker and use object storage instead of local disk.
