# Evaluation Report

## Automated engineering metrics

Run:

```powershell
python -m pytest --cov=app --cov-report=term-missing --cov-fail-under=80
python -m ruff check .
python -m mypy app
```

The repository currently includes unit tests for all document processors,
Google Vision response parsing, invoice context/prompt/schema/validation,
repositories, services, routers, and API-boundary integration tests.

## Dataset metrics

Accuracy, precision/recall, median processing time, success rate, and failure
rate must be measured with a labeled dataset containing representative digital
PDFs, scanned PDFs, images, XLS/XLSX files, CSV files, malformed documents,
and duplicate invoices. The application does not fabricate these metrics;
record them here after running the labeled evaluation set.

Recommended measurements:

- field-level exact-match accuracy for every invoice field;
- line-item precision and recall;
- document success/partial/failure rate;
- p50/p95 processing time by file type and page count;
- OCR confidence distribution;
- manual-review rate and duplicate-detection precision.

The repository includes a reproducible JSONL evaluator:

```powershell
python scripts/evaluate_extractions.py path\to\labeled-results.jsonl
```

Each JSONL record must contain `expected`, `actual`, and `status` objects. The
evaluator reports exact normalized field accuracy, line-item precision/recall,
and completed/failed rates. Processing latency and OCR confidence should be
collected by the worker when the labeled run is executed.
