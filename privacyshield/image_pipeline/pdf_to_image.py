"""
pdf_to_image.py
---------------
Converts a single PDF page to a PIL Image for OCR processing.
Uses pypdfium2 — no poppler needed, works on Windows out of the box.
"""

from __future__ import annotations
from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium


def pdf_page_to_image(pdf_path: str | Path, page_num: int, dpi: int = 200) -> Image.Image:
    """
    Convert one page of a PDF to a PIL Image.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF file.
    page_num : int
        0-based page index.
    dpi : int
        Resolution (200 is good balance of speed vs quality).

    Returns
    -------
    PIL Image object of that page.
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_num]

    # Scale factor: dpi / 72 (PDF default is 72 dpi)
    scale = dpi / 72
    bitmap = page.render(scale=scale, rotation=0)
    image = bitmap.to_pil()

    pdf.close()
    return image