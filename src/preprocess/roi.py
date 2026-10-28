import cv2
import pyvips
import numpy as np
from dataclasses import dataclass


_ROI_PADDING = 100


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    w: int
    h: int


def detect_roi(img_path: str, orig_w: int, orig_h: int) -> ROI:
    THUMB_SIZE = 1024
    thumb = pyvips.Image.thumbnail(img_path, THUMB_SIZE)
    scale_x = orig_w / thumb.width
    scale_y = orig_h / thumb.height

    blue = thumb[2] if thumb.bands >= 3 else thumb

    arr = np.ndarray(
        buffer=blue.write_to_memory(),
        dtype=np.uint8,
        shape=(blue.height, blue.width),
    )
    del thumb, blue  # free thumbnail immediately

    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = thresh > 0

    print(f"{img_path}: porcentaje máscara = {mask.mean():.3f}")

    rows_any = mask.any(axis=1)
    cols_any = mask.any(axis=0)

    if not rows_any.any():
        print(
            f"{img_path}: no ROI found. Falling back to full image."
        )
        return ROI(0, 0, orig_w, orig_h)

    row_min = int(np.argmax(rows_any))
    row_max = int(len(rows_any) - 1 - np.argmax(rows_any[::-1]))
    col_min = int(np.argmax(cols_any))
    col_max = int(len(cols_any) - 1 - np.argmax(cols_any[::-1]))

    x  = max(0,      int(col_min * scale_x) - _ROI_PADDING)
    y  = max(0,      int(row_min * scale_y) - _ROI_PADDING)
    x2 = min(orig_w, int((col_max + 1) * scale_x) + _ROI_PADDING)
    y2 = min(orig_h, int((row_max + 1) * scale_y) + _ROI_PADDING)

    roi = ROI(x=x, y=y, w=x2 - x, h=y2 - y)
    print(f"{img_path}: detected ROI -> {roi}")
    return roi