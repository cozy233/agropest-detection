"""YOLO-format annotation parsing, independent of the ML runtime."""

import math

CLASS_NAMES = (
    "Ants", "Bees", "Beetles", "Caterpillars", "Earthworms", "Earwigs",
    "Grasshoppers", "Moths", "Slugs", "Snails", "Wasps", "Weevils",
)


def parse_label(line, width, height):
    """Return a torchvision label (1..12) and clipped pixel xyxy box.

    A blank line returns None. Malformed annotations raise ValueError rather
    than being silently skipped or replaced with another image.
    """
    if not line.strip():
        return None
    values = line.split()
    if len(values) != 5:
        raise ValueError("Expected class_id x_center y_center width height")
    cls, xc, yc, bw, bh = map(float, values)
    if not all(math.isfinite(x) for x in (cls, xc, yc, bw, bh)):
        raise ValueError("Annotation contains a non-finite value")
    if not cls.is_integer() or not 0 <= cls < len(CLASS_NAMES):
        raise ValueError("Class id must be an integer from 0 to 11")
    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
        raise ValueError("Expected normalized coordinates and positive box size")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    box = [max(0.0, (xc - bw / 2) * width),
           max(0.0, (yc - bh / 2) * height),
           min(float(width), (xc + bw / 2) * width),
           min(float(height), (yc + bh / 2) * height)]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Box has no positive area after clipping")
    return int(cls) + 1, box
