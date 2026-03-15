"""
image_redactor.py
-----------------
Draws solid black boxes over PII regions on a PIL Image.
Takes OCR bounding boxes as input and blacks them out.
"""

from __future__ import annotations
from PIL import Image, ImageDraw


def redact_regions(image: Image.Image, regions: list[dict]) -> Image.Image:
    """
    Draw black boxes over all given bounding box regions on the image.

    Parameters
    ----------
    image : PIL Image
        The original page image.
    regions : list of dicts
        Each dict must have a "bbox" key with x, y, w, h values.
        These come directly from ocr_engine.extract_text_with_coords().

    Returns
    -------
    PIL Image with black boxes drawn over all PII regions.
    """
    # Work on a copy — never modify the original
    redacted = image.copy()
    draw = ImageDraw.Draw(redacted)

    for region in regions:
        bbox = region["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

        # Draw filled black rectangle over the region
        draw.rectangle(
            [x, y, x + w, y + h],
            fill="black"
        )

    return redacted


def redact_full_image(image: Image.Image) -> Image.Image:
    """
    Blur the entire image — used for photos/faces with no text.

    Parameters
    ----------
    image : PIL Image

    Returns
    -------
    PIL Image with gaussian blur applied.
    """
    from PIL import ImageFilter
    return image.filter(ImageFilter.GaussianBlur(radius=15))