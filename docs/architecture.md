# Backend Architecture

## Request flow

1. Upload validates extension, signature, integrity, and per-file/batch limits,
   writes the source to durable mounted storage, and inserts metadata.
2. Upload inserts a MongoDB job. Unique active-document keys serialize work;
   upload metadata is the outbox for recovering an enqueue interrupted by a
   process restart.
3. The independent worker claims jobs atomically, renews expiring leases, and
   records each attempt. Workers recover expired leases after crashes.
4. The PDF, image, XLS/XLSX, or CSV pipeline emits one page or bounded row
   chunk at a time. Parsers run on dedicated threads; each unit is persisted
   immediately and processing progress is updated.
5. When a document is ready and OpenAI is configured, the worker groups
   continuation pages and invoice-register rows into bounded candidates. It
   records uncertain boundaries for review, calls OpenAI, validates field
   evidence and arithmetic, checks supplier-scoped duplicate identities, and
   stores results.
6. Keyword search reads the MongoDB text index. Semantic search uses a bounded
   streaming cosine scan locally or Atlas Vector Search when configured.
7. Reconciliation compares invoice results from any two processed documents,
   reports missing/ambiguous matches and field/line-item differences, and
   persists a retrievable report.

## Collections

- `documents`: file metadata, storage path, status, progress, pages, and errors.
- `jobs`: durable work queue, lease/attempt state, outcomes, and audit events.
- `extraction_results`: source pages/sheets/CSV chunks, text, OCR, rows, and
  source references; a text index supports document-content search.
- `processing_history`: each document-processing attempt and outcome.
- `invoice_candidates`: invoice groups, source unit IDs, and review state.
- `invoice_extractions`: validated invoices, field evidence, corrections and
  correction history, model/prompt metadata, source fingerprints, and optional
  embeddings.
- `invoice_identities`: atomic supplier/invoice-number duplicate reservations.
- `reconciliations`: persisted two-document comparison reports.

## Deployment and reliability

Run one or more API containers and separate `python -m app.worker` processes.
MongoDB unique indexes, atomic job claims, lease heartbeats, retries, and
per-unit checkpoints provide durable at-least-once processing. Extraction IDs,
result unit IDs, and candidate IDs are deterministic across job retries.
Container deployments authenticate MongoDB using a least-privilege app user;
source uploads use a shared named volume for API/worker access. Set an API key,
restrict trusted hosts, enable TLS, and put shared rate limiting at the gateway
when running multiple API replicas.

The local default is a single shared API key and process-local request limit;
it does not provide per-user roles or a distributed rate-limit counter. For a
multi-tenant product, add tenant ownership checks before exposing documents.
