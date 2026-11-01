# Backend Walkthrough Recording Guide

This is a repeatable recording script for a backend-only demo. Start with
`docker compose up --build`, then use a second terminal. Configure a demo
invoice with clear text; do not show API keys in the recording.

1. Upload a PDF and XLSX with the upload endpoint. Show the returned document IDs.
2. Poll `/api/v1/documents/{document_id}/status` until processing completes; show page progress and processing history.
3. Inspect `/api/v1/documents/{document_id}/units` and `/candidates` to show page/cell references and invoice grouping.
4. Queue invoice extraction, then poll `/api/v1/jobs/{job_id}` and retrieve its extraction result.
5. Show extracted fields, evidence, review flags, and validation errors.
6. Patch one field. Show the correction audit and recalculated review status.
7. Compare the PDF with its spreadsheet at `/api/v1/reconciliations`; show matched and mismatched values.
8. Search a supplier, then show a corrupted file failure and a retry.

Capture the recording outside the repository so the video excludes secrets and
customer documents. The recording itself is an external submission artifact;
this file does not claim that a video was made.
