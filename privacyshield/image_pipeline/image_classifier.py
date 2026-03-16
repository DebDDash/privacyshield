"""
image_classifier.py
-------------------
Classifies an image region as:
  - "photo"        : face/logo, no readable text → blur it
  - "scanned_text" : document scan, has text → black box the text
  - "id_card"      : has BOTH face AND text → blur face + black box text

Method:
  1. Run OCR to check for text
  2. Run OpenCV face detection
  3. Classify based on combination
"""

from __future__ import annotations
import numpy as np
from PIL import Image

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_TEXT_LENGTH  = 10   # min total chars to count as "has text"
MIN_TEXT_REGIONS = 2    # min OCR blocks to count as "has text"
MIN_CONFIDENCE   = 0.6  # ignore low-confidence OCR results

# ── Singletons ────────────────────────────────────────────────────────────────
_ocr          = None
_face_cascade = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
    return _ocr


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        import cv2
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


# ── Detection helpers ─────────────────────────────────────────────────────────

def _detect_text(image: Image.Image) -> dict:
    """Run OCR and return whether meaningful text was found."""
    try:
        img_array = np.array(image.convert("RGB"))
        result = _get_ocr().ocr(img_array, cls=True)

        if not result or not result[0]:
            return {"has_text": False, "regions": [], "total_chars": 0}

        regions = []
        total_chars = 0
        for line in result[0]:
            text = line[1][0].strip()
            confidence = line[1][1]
            if confidence > MIN_CONFIDENCE and len(text) > 1:
                regions.append({"text": text, "confidence": confidence})
                total_chars += len(text)

        has_text = (
            len(regions) >= MIN_TEXT_REGIONS and
            total_chars  >= MIN_TEXT_LENGTH
        )
        return {"has_text": has_text, "regions": regions, "total_chars": total_chars}

    except Exception as e:
        print(f"[image_classifier] OCR failed: {e}")
        return {"has_text": False, "regions": [], "total_chars": 0}


def _detect_face(image: Image.Image) -> bool:
    """Return True if at least one face is detected in the image."""
    try:
        import cv2
        gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        faces = _get_face_cascade().detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        return len(faces) > 0
    except Exception as e:
        print(f"[image_classifier] Face detection failed: {e}")
        return False


# ── Main classifier ───────────────────────────────────────────────────────────

def classify_image(image: Image.Image) -> dict:
    """
    Classify an image and return what action the redactor should take.

    Returns
    -------
    {
        "type"        : "photo" | "scanned_text" | "id_card",
        "has_text"    : bool,
        "has_face"    : bool,
        "text_regions": list,   ← OCR results if text found
        "total_chars" : int,
        "action"      : "blur" | "blackbox_text" | "blur_and_blackbox"
    }
    """
    text_result = _detect_text(image)
    has_face    = _detect_face(image)
    has_text    = text_result["has_text"]

    if has_text and has_face:
        image_type = "id_card"
        action     = "blur_and_blackbox"
    elif has_text:
        image_type = "scanned_text"
        action     = "blackbox_text"
    else:
        image_type = "photo"
        action     = "blur"

    return {
        "type"        : image_type,
        "has_text"    : has_text,
        "has_face"    : has_face,
        "text_regions": text_result["regions"],
        "total_chars" : text_result["total_chars"],
        "action"      : action,
    }