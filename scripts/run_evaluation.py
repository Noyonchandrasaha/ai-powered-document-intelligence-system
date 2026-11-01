"""Measure real parser timings and optionally live OCR + LLM extraction accuracy.

--live sends the synthetic corpus to configured Google Vision/OpenAI services.
Offline runs explicitly leave AI accuracy unmeasured; mocks are never scored.
"""

import argparse
import asyncio
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import get_settings
from app.features.document_extraction.data_processing.ocr.google_vision import GoogleVisionOCR
from app.features.document_extraction.data_processing.pipeline import process_document
from app.features.document_extraction.validators import validate_document
from app.features.invoices.context import build_invoice_context
from app.features.invoices.extractor import OpenAIInvoiceExtractor
from app.features.invoices.validators import validate_invoice_extraction
from scripts.evaluate_extractions import evaluate


async def run(manifest: Path, live: bool) -> dict[str, Any]:
    settings = get_settings()
    records = []
    if live and (
        not settings.openai_api_key or not Path(settings.google_vision_credentials_file).is_file()
    ):
        raise ValueError("Live evaluation needs OpenAI and Google Vision credentials.")
    async with AsyncExitStack() as stack:
        ocr = (
            await stack.enter_async_context(
                GoogleVisionOCR(settings.google_vision_credentials_file)
            )
            if live
            else None
        )
        client = (
            await stack.enter_async_context(AsyncOpenAI(api_key=settings.openai_api_key))
            if live
            else None
        )
        extractor = OpenAIInvoiceExtractor(client, model=settings.openai_model) if client else None
        for line in manifest.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            path = (manifest.parent / entry["file"]).resolve()
            if not path.is_relative_to(manifest.parent.resolve()):
                raise ValueError("Dataset files must remain inside the manifest directory.")
            started = perf_counter()
            record: dict[str, Any] = {
                "file": entry["file"],
                "file_type": path.suffix,
                "expected_rejection": entry.get("expected_rejection", False),
            }
            if entry.get("requires_ocr") and not live:
                record.update(status="skipped", reason="Live OCR credentials required")
                records.append(record)
                continue
            try:
                kind, _ = await asyncio.to_thread(validate_document, path, path.name)
                units = [
                    dict(unit) async for unit in process_document("evaluation", path, kind, ocr)
                ]
                record["units"] = len(units)
                record["status"] = (
                    "completed"
                    if units and all(u["status"] == "completed" for u in units)
                    else "partial"
                    if any(u["status"] != "failed" for u in units)
                    else "failed"
                )
                if extractor and not record["expected_rejection"]:
                    context = build_invoice_context("evaluation", units)
                    ai = await extractor.extract(context)
                    actual = validate_invoice_extraction("evaluation", ai, context)
                    record.update(
                        expected=entry["expected"],
                        actual=actual.invoice.model_dump(mode="json"),
                        extraction_status=actual.status.value,
                    )
                record["correct_rejection"] = False
            except Exception as exc:
                record.update(
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    correct_rejection=record["expected_rejection"],
                )
            record["duration_seconds"] = round(perf_counter() - started, 6)
            records.append(record)
    valid = [row for row in records if not row["expected_rejection"] and row["status"] != "skipped"]
    scored = [row for row in records if "actual" in row]
    return {
        "mode": "live_ocr_llm" if live else "offline_real_parsers",
        "synthetic_dataset": True,
        "records": records,
        "pipeline_metrics": evaluate(valid),
        "ai_accuracy": evaluate(scored) if scored else None,
        "correct_rejections": sum(bool(row.get("correct_rejection")) for row in records),
        "skipped": sum(row["status"] == "skipped" for row in records),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("docs/evaluation-run.jsonl"))
    args = parser.parse_args()
    report = asyncio.run(run(args.manifest, args.live))
    args.output.write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
