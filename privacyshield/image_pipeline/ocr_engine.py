"""
ocr_engine.py
-------------
Runs PaddleOCR on a PIL Image and returns detected text
with exact pixel bounding boxes.
"""

from __future__ import annotations
import os
import numpy as np
from PIL import Image

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

_ocr = None

def _get_ocr():
    """Load PaddleOCR once and reuse across calls."""
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
    return _ocr


def extract_text_with_coords(image: Image.Image) -> list[dict]:
    """
    Run OCR on a PIL Image and return text regions with bounding boxes.
    Returns list of dicts with "text", "confidence", "bbox" (x,y,w,h).
    Coordinates are in pixels, origin at top-left.
    """
    img_array = np.array(image.convert("RGB"))
    ocr = _get_ocr()
    result = ocr.ocr(img_array, cls=True)

    regions = []
    if not result or not result[0]:
        return regions

    for line in result[0]:
        polygon    = line[0]
        text       = line[1][0]
        confidence = line[1][1]

        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        regions.append({
            "text": text,
            "confidence": confidence,
            "bbox": {
                "x": int(min(xs)),
                "y": int(min(ys)),
                "w": int(max(xs) - min(xs)),
                "h": int(max(ys) - min(ys)),
            }
        })

    return regions