# Application Walkthrough Recording Guide

This is a recording script for the browser application. Start the API and
worker (or run `docker compose up --build`), then open `/ui/`. Configure a demo
invoice with clear text; do not show API keys in the recording.

1. In the dashboard, upload a PDF and XLSX. Show upload progress and document status.
2. Open document details; show processed pages/sheets and invoice groups.
3. Open invoice review; show extracted fields, line items, source evidence,
   validation errors, and review status.
4. Correct one field and show the updated status/audit details.
5. Search by supplier, then compare the PDF and spreadsheet in cross verification.
6. Upload a corrupted test PDF and show its failure status and retry control.

Capture the recording outside the repository so the video excludes secrets and
customer documents. The recording itself is an external submission artifact;
this file does not claim that a video was made.
