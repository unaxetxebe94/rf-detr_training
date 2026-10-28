import cv2
import pyvips
import numpy as np
from dataclasses import dataclass


_ROI_PADDING = 100
_THRESHOLD = 10


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

    arr = np.ndarray(buffer=blue.write_to_memory(), dtype=np.uint8, shape=(blue.height, blue.width))

    del blue

    _, thresh = cv2.threshold(arr,  _THRESHOLD,  255, cv2.THRESH_BINARY)

    # Eliminar ruido pequeño
    kernel = np.ones((5, 5), np.uint8)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    mask = thresh > 0

    print(f"{img_path}: porcentaje máscara = {mask.mean():.3f}")

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    if num_labels <= 1:
        print(f"{img_path}: no ROI found. Falling back to full image.")
        return ROI(0, 0, orig_w, orig_h)

    # Ignorar fondo (label 0) y quedarse con la componente más ancha
    widest_label = max(range(1, num_labels), key=lambda label: stats[label, cv2.CC_STAT_WIDTH])

    left = stats[widest_label, cv2.CC_STAT_LEFT]
    top = stats[widest_label, cv2.CC_STAT_TOP]
    width = stats[widest_label, cv2.CC_STAT_WIDTH]
    height = stats[widest_label, cv2.CC_STAT_HEIGHT]
    area = stats[widest_label, cv2.CC_STAT_AREA]

    print(f"{img_path}: widest component (width={width}, height={height}, area={area})")

    x = max(0, int(left * scale_x) - _ROI_PADDING)
    y = max(0, int(top * scale_y) - _ROI_PADDING)
    x2 = min(orig_w, int((left + width) * scale_x) + _ROI_PADDING)
    y2 = min(orig_h, int((top + height) * scale_y) + _ROI_PADDING)

    roi = ROI(x=x, y=y, w=x2 - x, h=y2 - y)

    print(f"{img_path}: detected ROI -> {roi}")

    return roi