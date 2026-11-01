# MongoDB Schema

## `documents`

```text
document_id: string, unique
filename: string; file_type: string; file_size: integer
status: pending | processing | completed | partial | failed
created_at, updated_at: datetime
total_pages: integer|null; processed_units: integer
storage_path: string; error_message: string|null
```

Indexes: unique `document_id`; `(status, created_at)` for worker/outbox scans.

## `jobs`

```text
job_id: string, unique; kind: process | extract
document_id: string; payload: object; status: queued | running | completed | failed
attempts: integer; available_at, lease_until, created_at, started_at: datetime
active_key: document_id while queued/running; owner: worker id
result: object|null; error: string|null; history: object[]
```

Indexes: unique partial `active_key`; `(status, available_at)` for atomic
claims; `(document_id, created_at)` for inspection.

## `extraction_results`

```text
document_id: string; unit_id: string; kind: page | sheet_chunk | csv_chunk
status: completed | partial | failed
text/native_text/ocr_text: string; search_text: indexed string
page_number/sheet_name/sheet_index/start_row/end_row: source location
rows/text_blocks/words: structured source data; metadata: object
error: string|null
```

Unique index: `(document_id, unit_id)`. Full-text index: `search_text`.

## `processing_history`

```text
document_id: string; status: processing | completed | partial | failed
processed_units: integer; started_at: datetime; completed_at: datetime|null
error_message: string|null
```

Index: `(document_id, started_at desc)`.

## `invoice_candidates`

```text
candidate_id: string; job_id: string; document_id: string
unit_ids: string[]; invoice_hint: string|null; requires_review: boolean
status: pending | failed; error: string|null
```

## `invoice_extractions`

Validated response fields plus `extraction_id`, `model_name`, `prompt_version`,
`source_units`, SHA-256 `source_fingerprint`, optional `embedding: float[]`,
`revision: integer`, and append-only `correction_history`. Unique index on
`extraction_id`; lookup index `(document_id, extracted_at desc, extraction_id
desc)`.

## `invoice_identities`

```text
identity: SHA-256(supplier + invoice_number), unique
document_id: string
```

Atomic reservations prevent concurrent uploads with the same supplier and
invoice number from both appearing as new invoices.

## `reconciliations`

```text
reconciliation_id: string, unique; document_ids: [string, string]
amount_tolerance: string; status: matched | needs_review
counts: object; rows: object[]; created_at: datetime
```
