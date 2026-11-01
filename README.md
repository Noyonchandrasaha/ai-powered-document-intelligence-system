# AI-Powered Document Intelligence Backend

FastAPI backend for processing business documents, extracting invoice data with
OpenAI, validating the result against source evidence, and storing searchable
results in MongoDB. A plain HTML/CSS/JavaScript review dashboard is served by
FastAPI at `/ui/`.

## Architecture

```text
Upload API
   |
   v
Local file storage + Mongo document metadata
   |
   v
   MongoDB durable job queue
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

The backend stores metadata, extracted units, processing history, and validated invoice results in MongoDB. A durable MongoDB lease queue keeps OCR and AI work outside API requests. The separate worker claims jobs atomically, renews leases, records attempts, and recovers abandoned work after restarts. Processing checkpoints are saved per page, sheet, or row chunk.

## Local setup

1. Create and activate a Python 3.12 virtual environment (Python 3.13 also works in the tested local environment).

   ```powershell
   py -3.12 -m venv .venv
   .venv\\Scripts\\Activate.ps1
   ```
2. Install dependencies:

   ```powershell
   python -m pip install -e ".[dev]"
   ```

3. Copy `.env.example` to `.env` and configure your existing MongoDB connection
   string (`APP_MONGO_URI`) and database name. Configure OpenAI for invoice AI
   extraction and Google Vision for scanned/image OCR; digital PDF and
   spreadsheet parsing can be tested without those external services.
4. Start the API and worker:

   ```powershell
   uvicorn app.main:app --reload
   # In a second terminal, activate .venv and run:
   python -m app.worker
   ```

5. Open `http://localhost:8000/ui/` for the browser dashboard. The dashboard
   supports multi-file uploads, document processing monitoring, source-unit
   inspection, invoice extraction and review, corrections, search, and
   document-to-document reconciliation. If `APP_API_KEY` is configured, use
   **Workspace admin → API settings** to enter it in the browser.

Or run the API and worker with Docker Compose, using the MongoDB server in
`APP_MONGO_URI` (Compose does not start a MongoDB container):

```powershell
docker compose up --build
```

Set `APP_MONGO_URI` to your MongoDB connection string in `.env`. The API and
worker both connect to this external MongoDB server; Compose does not create a
MongoDB container. Set a random
`APP_API_KEY` (at least 32 characters) only if you want API authentication.
Uploads persist in a named Docker volume. For scanned documents, place Google
Vision credentials at `secrets/google-vision.json`. For local runs, set
`APP_GOOGLE_VISION_CREDENTIALS_FILE` to the credential JSON path in `.env`.

Never commit `.env`, service-account credentials, or API keys.

## API endpoints

### Documents

- `POST /api/v1/documents/upload` — upload PDF, JPG, PNG, TIFF, XLSX, XLS, or CSV files.
- `GET /api/v1/documents/{document_id}` — document metadata.
- `GET /api/v1/documents/{document_id}/status` — processing status and errors.
- `GET /api/v1/documents/{document_id}/history` — processing attempts.
- `POST /api/v1/documents/{document_id}/reprocess` — queue failed/partial processing again.
- `GET /api/v1/documents/search` — keyword/status filtering.
- `GET /api/v1/documents/{document_id}/units` ? inspect extracted pages, sheets, and rows.
- `GET /api/v1/documents/{document_id}/candidates` ? inspect invoice groups.
- `DELETE /api/v1/documents/{document_id}` — delete metadata and local source file.

### Invoice extraction

- `POST /api/v1/invoices/documents/{document_id}/extract` — queue invoice extraction and return a job ID.
- `GET /api/v1/jobs/{job_id}` ? poll processing and extraction jobs.
- `POST /api/v1/jobs/{job_id}/retry` ? retry failed jobs.
- `GET /api/v1/invoices/extractions/{extraction_id}` — retrieve one result.
- `GET /api/v1/invoices/documents/{document_id}/latest` — retrieve latest result.
- `GET /api/v1/invoices/search` — search invoice number, supplier, customer, or document ID.
- `GET /api/v1/invoices/search/semantic` — embedding-based invoice search.
- `POST /api/v1/reconciliations` ? cross-check invoices from two documents.
- `GET /api/v1/reconciliations/{reconciliation_id}` ? retrieve a comparison report.
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

Set `APP_SEMANTIC_SEARCH_ENABLED=true` to create embeddings. Local search streams stored vectors and retains only the top results; `APP_ATLAS_VECTOR_INDEX` selects MongoDB Atlas Vector Search. Explicit invoice numbers group continuation pages and invoice register rows become separate candidates. Ambiguous grouping requires manual review; use the source-unit API to select a 1,000-page invoice when the source has no clear boundaries.

## Quality checks

```powershell
python -m pytest --cov=app --cov-report=term-missing --cov-fail-under=80
python -m ruff check app tests scripts
python -m mypy app
python scripts/generate_evaluation_dataset.py
python scripts/run_evaluation.py evaluation-data/manifest.jsonl --output docs/evaluation-run.jsonl
```

Tests are organized under `tests/unit` and `tests/integration`. Integration
tests use controlled Mongo/OpenAI adapters and do not require live external
services.

Latest local engineering checks (2026-11-01): **254 tests passed**, **81.04%
coverage**, Ruff passed, and mypy passed (43 source files). The reproducible
synthetic parser benchmark completed 4/4 valid non-OCR documents (digital PDF,
two-sheet XLSX, XLS, CSV); both corrupt/misleading-extension samples were
correctly rejected. In the latest run parser p50 was 4.7 ms and p95 was 25.2
ms. Six scan/image samples were skipped because Google Vision credentials
were not configured. Consequently live OCR performance and OpenAI invoice
field/line-item accuracy remain unmeasured; these smoke results do not
represent production throughput or labeled-corpus accuracy. See
[`docs/evaluation.md`](docs/evaluation.md) and [`docs/evaluation-run.jsonl`](docs/evaluation-run.jsonl).
Automated check results are also saved in [`docs/test-results.md`](docs/test-results.md).

## Production notes

- Set `APP_ENVIRONMENT=production`, a 32-character `APP_API_KEY`, and restricted `APP_ALLOWED_HOSTS`.
- Keep `APP_OPENAI_API_KEY`, Mongo passwords, and Google service-account credentials outside git.
- Mount Google Vision credentials at `/app/secrets/google-vision.json` when using Docker.
- Put the API behind TLS and a gateway that shares rate limits across replicas.
- Run additional worker containers for more processing capacity.
- Run one or more `worker` instances separately from API replicas.
- Run extra worker containers for more processing capacity.
- Replace local volume storage with object storage for multi-host deployments.

## Submission status and walkthrough

The backend, browser UI, schema/architecture documentation, prompts/disclosure,
synthetic parser evaluation, and automated engineering checks are included.
The walkthrough recording is not included; `docs/backend-walkthrough.md` is a
recording guide only. Live OCR/LLM accuracy still requires configured service
credentials and a human-labeled evaluation corpus.
