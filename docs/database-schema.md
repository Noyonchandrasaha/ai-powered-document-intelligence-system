# Database Schema

## `documents`

```text
document_id: string, unique
filename: string
file_type: string
file_size: integer
status: pending | processing | completed | partial | failed
created_at: datetime
updated_at: datetime
total_pages: integer|null
storage_path: string
error_message: string|null
```

## `extraction_results`

```text
document_id: string
unit_id: string
kind: page | sheet_chunk | csv_chunk
status: completed | partial | failed
text/native_text/ocr_text: string
page_number/sheet_name/sheet_index: optional source location
rows/text_blocks/words: structured source data
metadata: object
error: string|null
```

Unique index: `(document_id, unit_id)`.

## `processing_history`

```text
document_id: string
status: processing | completed | partial | failed
processed_units: integer
started_at: datetime
completed_at: datetime|null
error_message: string|null
```

## `invoice_extractions`

The validated `InvoiceExtractionResponse` fields are stored with
`extraction_id`, `model_name`, `prompt_version`, `source_units`, and creation
metadata. When semantic search is enabled, an `embedding: float[]` field is
also stored. `extraction_id` is unique; document results are indexed by
document and extraction time.
