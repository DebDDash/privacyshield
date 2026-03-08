"""
pdf_rebuilder.py
================
PURPOSE:
    Takes the original PDF and draws black redaction boxes over
    all PII spans using the bounding box coordinates from pipeline.py.

    Does NOT recreate the PDF from scratch — overlays black rectangles
    on the original page. This preserves fonts, images, layout exactly.

METHOD:
    Uses PyMuPDF (fitz) to:
    1. Open original PDF
    2. For each redaction box → draw filled black rectangle
    3. Apply redactions (permanently removes underlying content)
    4. Save redacted PDF to output path

COORDINATE CONVERSION:
    extractor.py uses pdfplumber top-origin coordinates:
        y=0 at TOP of page, increases downward
    PyMuPDF uses PDF bottom-origin coordinates:
        y=0 at BOTTOM of page, increases upward
    Conversion: pdf_y = page_height - plumber_y

DEPENDENCIES:
    pip install pymupdf
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    logger.warning("PyMuPDF not installed. Run: pip install pymupdf")


def _plumber_to_fitz(bbox: dict, page_height: float) -> tuple:
    """
    Convert pdfplumber bbox to PyMuPDF rect.

    Both pdfplumber and PyMuPDF use top-origin coordinates
    (y=0 at top, increases downward) so NO flipping needed.
    We just add padding to fully cover the text.

    Args:
        bbox: {"x0": float, "y0": float, "x1": float, "y1": float}
        page_height: unused, kept for API consistency

    Returns:
        fitz.Rect compatible tuple (x0, y0, x1, y1)
    """
    padding = 1.5
    return (
        bbox["x0"] - padding,
        bbox["y0"] - padding,
        bbox["x1"] + padding,
        bbox["y1"] + padding,
    )


def rebuild_pdf(
    original_pdf_path: str,
    pipeline_result: dict,
    output_path: str,
) -> str:
    """
    Draw black redaction boxes over all PII spans in the original PDF.

    Args:
        original_pdf_path: Path to the original unredacted PDF.
        pipeline_result: Output dict from pipeline.run_text_pipeline().
            Must contain "pages" list with "redaction_boxes" per page.
        output_path: Where to save the redacted PDF.

    Returns:
        output_path as string.

    Raises:
        ImportError: If PyMuPDF is not installed.
        FileNotFoundError: If original PDF not found.
    """
    if not FITZ_AVAILABLE:
        raise ImportError("PyMuPDF required: pip install pymupdf")

    original_pdf_path = str(original_pdf_path)
    output_path = str(output_path)

    if not Path(original_pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {original_pdf_path}")

    doc = fitz.open(original_pdf_path)
    total_boxes = 0

    for page_result in pipeline_result["pages"]:
        page_num = page_result["page_number"]
        redaction_boxes = page_result.get("redaction_boxes", [])

        if not redaction_boxes:
            logger.debug(f"Page {page_num}: no redaction boxes, skipping")
            continue

        # PyMuPDF is 0-indexed
        page = doc[page_num - 1]
        page_height = page.rect.height

        for box in redaction_boxes:
            bbox = box["bbox"]
            entity_type = box["entity_type"]

            # Convert coordinates
            rect = _plumber_to_fitz(bbox, page_height)
            fitz_rect = fitz.Rect(rect)

            # Add redaction annotation (black fill, no border)
            page.add_redact_annot(fitz_rect, fill=(0, 0, 0))
            logger.debug(f"  Page {page_num}: redacting [{entity_type}] at {rect}")
            total_boxes += 1

        # Apply all redactions on this page — permanently removes underlying text
        page.apply_redactions()

    logger.info(f"Applied {total_boxes} redaction boxes across {len(doc)} pages")

    # Save redacted PDF
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()

    logger.info(f"Saved redacted PDF to: {output_path}")
    return output_path


def rebuild_pdf_with_labels(
    original_pdf_path: str,
    pipeline_result: dict,
    output_path: str,
    show_labels: bool = True,
) -> str:
    """
    Like rebuild_pdf but optionally shows [TOKEN_ID] label in white text
    on top of the black redaction box.

    Useful for reviewing what was redacted without exposing original values.

    Args:
        original_pdf_path: Path to original PDF.
        pipeline_result: Output from run_text_pipeline().
        output_path: Output path for redacted PDF.
        show_labels: If True, print [TOKEN_ID] in white on black box.

    Returns:
        output_path as string.
    """
    if not FITZ_AVAILABLE:
        raise ImportError("PyMuPDF required: pip install pymupdf")

    original_pdf_path = str(original_pdf_path)
    output_path = str(output_path)

    if not Path(original_pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {original_pdf_path}")

    # Build reverse map: original_text → token_id
    token_map = pipeline_result.get("token_map", {})
    value_to_token = {v: k for k, v in token_map.items()}

    doc = fitz.open(original_pdf_path)

    for page_result in pipeline_result["pages"]:
        page_num = page_result["page_number"]
        redaction_boxes = page_result.get("redaction_boxes", [])

        if not redaction_boxes:
            continue

        page = doc[page_num - 1]
        page_height = page.rect.height

        for box in redaction_boxes:
            bbox = box["bbox"]
            original_text = box.get("text", "")

            rect = _plumber_to_fitz(bbox, page_height)
            fitz_rect = fitz.Rect(rect)

            # Add black redaction box
            page.add_redact_annot(fitz_rect, fill=(0, 0, 0))

        page.apply_redactions()

        if show_labels:
            # Draw white token labels on top of black boxes AFTER redaction
            for box in redaction_boxes:
                bbox = box["bbox"]
                original_text = box.get("text", "")
                # Try full text first, then stripped value (for medical conditions)
                stripped = original_text.split(":", 1)[-1].strip() if ":" in original_text else original_text
                token_id = value_to_token.get(stripped, value_to_token.get(original_text, "REDACTED"))
                label = f"[{token_id}]"

                rect = _plumber_to_fitz(bbox, page_height)
                fitz_rect = fitz.Rect(rect)

                # White text on black box
                page.insert_textbox(
                    fitz_rect,
                    label,
                    fontsize=6,
                    color=(1, 1, 1),  # white
                    align=1,          # center
                )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()

    logger.info(f"Saved labeled redacted PDF to: {output_path}")
    return output_path
