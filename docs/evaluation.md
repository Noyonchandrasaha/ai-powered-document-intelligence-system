# Evaluation Report

## Local synthetic benchmark (2026-11-01)

Command: `python scripts/run_evaluation.py evaluation-data/manifest.jsonl`.
This repeatable smoke benchmark uses synthetic documents and local parsers, not
customer data. Four of four valid non-OCR documents completed: digital PDF,
XLSX (two sheets), XLS, and CSV. Parser success was 100%; one run measured
34.4 ms p50 / 122.4 ms p95. Both invalid files (corrupt PDF and disguised PNG)
were rejected. Six scan/image cases were skipped without Google Vision
credentials. Live OCR and LLM field accuracy are therefore unmeasured; these
offline figures are not end-to-end invoice accuracy or production throughput.
Per-document results are in `evaluation-run.jsonl`.

Recreate the synthetic corpus and run the real parsers:

```powershell
python scripts/generate_evaluation_dataset.py
python scripts/run_evaluation.py evaluation-data/manifest.jsonl
```

For live OCR and AI measurements, configure Google Vision and OpenAI:

```powershell
python scripts/run_evaluation.py evaluation-data/manifest.jsonl --live
```

The generated manifest covers digital and scanned PDFs, JPG, PNG, TIFF,
multipage TIFF, poor-quality images, multi-sheet XLSX, XLS, CSV, corrupted
PDFs, and misleading extensions. Use a human-labeled business corpus for
submission-grade field accuracy, line-item precision/recall, and real document
success/failure measurements. Do not score mocked output as measured accuracy.

## Automated engineering checks

```powershell
python -m pytest --cov=app --cov-report=term-missing --cov-fail-under=80
python -m ruff check app tests scripts
python -m mypy app
```

The suite includes unit tests for all parsers, OCR response handling, invoice
context, prompts, schemas, validation, repositories, services, and routes, plus
integration tests for file processing and backend workflows.

The JSONL evaluator also accepts `expected`, `actual`, and `status` records. It
reports normalized field accuracy, line-item precision/recall, success/failure
rates, and p50/p95 latency when duration samples are available:

```powershell
python scripts/evaluate_extractions.py path\to\labeled-results.jsonl
```
