"""Generate a deterministic synthetic labeled corpus; contains no customer data."""

import argparse
import csv
import json
from pathlib import Path

import pymupdf
import xlwt
from openpyxl import Workbook
from PIL import Image, ImageDraw, ImageFont

LABEL = {
    "invoice_number": "INV-2026-001",
    "invoice_date": "2026-10-01",
    "supplier_name": "Example Supplies Ltd",
    "customer_name": "Example Customer Ltd",
    "subtotal": "100.00",
    "tax": "10.00",
    "total": "110.00",
    "currency": "USD",
    "line_items": [
        {"description": "Office Paper", "quantity": "2", "unit_price": "50.00", "amount": "100.00"}
    ],
}
TEXT = """INVOICE
Invoice Number: INV-2026-001
Invoice Date: 2026-10-01
Supplier: Example Supplies Ltd
Customer: Example Customer Ltd
Currency: USD
Description | Quantity | Unit Price | Amount
Office Paper | 2 | 50.00 | 100.00
Subtotal: 100.00
Tax: 10.00
Total: 110.00
"""


def generate(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    entries = []

    def add(name, **kwargs):
        entries.append({"file": name, "expected": LABEL, "expected_rejection": False, **kwargs})

    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), TEXT, fontsize=12)
        pdf.save(directory / "digital.pdf")
    add("digital.pdf")
    image = Image.new("RGB", (1400, 1200), "white")
    ImageDraw.Draw(image).multiline_text(
        (60, 60), TEXT, fill="black", font=ImageFont.load_default(size=30), spacing=12
    )
    for extension in ("png", "jpg", "tiff"):
        image.save(directory / f"invoice.{extension}")
        add(f"invoice.{extension}", requires_ocr=True)
    image.save(directory / "multipage.tiff", save_all=True, append_images=[image])
    add("multipage.tiff", requires_ocr=True)
    with pymupdf.open() as pdf:
        pdf.new_page(width=700, height=600).insert_image(
            pymupdf.Rect(0, 0, 700, 600), filename=str(directory / "invoice.png")
        )
        pdf.save(directory / "scanned.pdf")
    add("scanned.pdf", requires_ocr=True)
    low_contrast = image.convert("L").point(lambda x: int(180 + x * 0.25))
    low_contrast.save(directory / "poor-quality.png")
    add("poor-quality.png", requires_ocr=True)
    rows = [[key, value] for key, value in LABEL.items() if key != "line_items"]
    rows += [["description", "quantity", "unit_price", "amount"], ["Office Paper", 2, 50, 100]]
    book = Workbook()
    for name in ("Invoice", "Second Sheet"):
        sheet = book.active if name == "Invoice" else book.create_sheet(name)
        sheet.title = name
        for row in rows:
            sheet.append(row)
    book.save(directory / "invoice.xlsx")
    add("invoice.xlsx")
    legacy = xlwt.Workbook()
    sheet = legacy.add_sheet("Invoice")
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            sheet.write(r, c, value)
    legacy.save(str(directory / "invoice.xls"))
    add("invoice.xls")
    with (directory / "invoice.csv").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    add("invoice.csv")
    (directory / "corrupt.pdf").write_bytes(b"%PDF-1.7\ninvalid document")
    add("corrupt.pdf", expected_rejection=True, expected={})
    (directory / "disguised.png").write_bytes(b"executable content")
    add("disguised.png", expected_rejection=True, expected={})
    manifest = directory / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("evaluation-data"))
    print(generate(parser.parse_args().directory))
