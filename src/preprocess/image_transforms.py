import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


ALL_AUGMENTATIONS = ["brightness", "contrast", "saturation", "hue", "blur"]


def shift_hue(image: Image.Image, degrees: float) -> Image.Image:
    arr = np.array(image, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    max_c = np.max(arr, axis=2)
    min_c = np.min(arr, axis=2)
    delta = max_c - min_c
    s = np.where(max_c > 0, delta / np.maximum(max_c, 0.0001), 0.0)
    v = max_c
    h = np.zeros_like(max_c)
    mask_r = (max_c == r) & (delta > 0)
    mask_g = (max_c == g) & (delta > 0)
    mask_b = (max_c == b) & (delta > 0)
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360
    h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120
    h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240
    h = (h + degrees) % 360
    hi = (h / 60).astype(int) % 6
    f = h / 60 - np.floor(h / 60)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    out = np.zeros_like(arr)
    for val, (rv, gv, bv) in enumerate([
        (v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)
    ]):
        mask = hi == val
        out[mask, 0] = rv[mask]
        out[mask, 1] = gv[mask]
        out[mask, 2] = bv[mask]
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))


def apply_transforms(
    image: Image.Image, transforms: list[str], rng: random.Random
) -> Image.Image:
    unknown = set(transforms) - set(ALL_AUGMENTATIONS)
    if unknown:
        raise ValueError(f"Aumentaciones desconocidas: {unknown}.")
    img = image.copy()
    try:
        for transform in transforms:
            if transform == "brightness":
                transformed = ImageEnhance.Brightness(img).enhance(rng.uniform(0.6, 1.4))
            elif transform == "contrast":
                transformed = ImageEnhance.Contrast(img).enhance(rng.uniform(0.6, 1.4))
            elif transform == "saturation":
                transformed = ImageEnhance.Color(img).enhance(rng.uniform(0.5, 1.5))
            elif transform == "hue":
                transformed = shift_hue(img, rng.uniform(-30.0, 30.0))
            else:
                transformed = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.0)))
            img.close()
            img = transformed
        return img
    except Exception:
        img.close()
        raise
