"""
Augmentator — Aumento de datos para datasets COCO con segmentaciones tipo polígono.

Dependencias:
    pip install pillow numpy

Uso básico:
    aug = Augmentator(
        images_dir="data/images",
        annotations_path="data/annotations.json",
        output_dir="data/augmented",
    )
    aug.run()
"""

import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any
import logging
from logger import get_logger

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ---------------------------------------------------------------------------
# Helpers geométricos
# ---------------------------------------------------------------------------

def _flip_polygons_horizontal(polygons: list[list[float]], width: int) -> list[list[float]]:
    """Refleja polígonos horizontalmente: x' = W - 1 - x."""
    result = []
    for seg in polygons:
        coords = []
        for i in range(0, len(seg), 2):
            coords.append(width - 1 - seg[i])   # x
            coords.append(seg[i + 1])            # y
        result.append(coords)
    return result


def _flip_polygons_vertical(polygons: list[list[float]], height: int) -> list[list[float]]:
    """Refleja polígonos verticalmente: y' = H - 1 - y."""
    result = []
    for seg in polygons:
        coords = []
        for i in range(0, len(seg), 2):
            coords.append(seg[i])                  # x
            coords.append(height - 1 - seg[i + 1]) # y
        result.append(coords)
    return result


def _rotate_polygons_90(polygons: list[list[float]], width: int, height: int,
                        times: int) -> tuple[list[list[float]], int, int]:
    """
    Rota polígonos 'times' veces 90° en sentido antihorario.
    Devuelve (polygons_nuevos, new_width, new_height).
    """
    for _ in range(times % 4):
        new_polys = []
        for seg in polygons:
            coords = []
            for i in range(0, len(seg), 2):
                x, y = seg[i], seg[i + 1]
                # 90° antihorario: (x, y) → (y, W-1-x)
                coords.append(y)
                coords.append(width - 1 - x)
            new_polys.append(coords)
        polygons = new_polys
        width, height = height, width  # las dimensiones se intercambian
    return polygons, width, height


def _recompute_bbox(segmentation: list[list[float]]) -> list[float]:
    """Calcula el bounding box [x, y, w, h] a partir de los polígonos."""
    all_x, all_y = [], []
    for seg in segmentation:
        all_x.extend(seg[0::2])
        all_y.extend(seg[1::2])
    x_min, y_min = min(all_x), min(all_y)
    x_max, y_max = max(all_x), max(all_y)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _recompute_area(segmentation: list[list[float]]) -> float:
    """Área aproximada mediante la fórmula de Shoelace para cada polígono."""
    total = 0.0
    for seg in segmentation:
        xs = seg[0::2]
        ys = seg[1::2]
        n = len(xs)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += xs[i] * ys[j]
            area -= xs[j] * ys[i]
        total += abs(area) / 2.0
    return total


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class Augmenter:
    """
    Aplica aumentaciones a un dataset COCO con segmentaciones poligonales y
    guarda las imágenes sintéticas junto con un nuevo archivo de anotaciones.

    Parámetros
    ----------
    images_dir : str
        Carpeta con las imágenes originales.
    annotations_path : str
        Ruta al archivo JSON en formato COCO.
    output_dir : str
        Carpeta de salida (se crean subdirectorios 'images/' y se genera
        'annotations.json').
    augmentations : list[str] | None
        Lista de aumentaciones a usar. Si es None se usan todas.
        Opciones: "hflip", "vflip", "rot90", "rot180",
                  "brightness", "contrast", "saturation", "hue", "blur"
    augmentations_per_image : int
        Cuántas imágenes aumentadas se generan por imagen original.
    max_transforms_per_sample : int
        Número máximo de transformaciones que se combinan en una sola imagen.
    brightness_range : tuple[float, float]
        Rango del factor de brillo (1.0 = sin cambio).
    contrast_range : tuple[float, float]
        Rango del factor de contraste.
    saturation_range : tuple[float, float]
        Rango del factor de saturación.
    hue_range : tuple[float, float]
        Rango de rotación del tono en grados (−180, 180).
    blur_radius_range : tuple[float, float]
        Rango del radio del filtro Gaussian blur.
    seed : int | None
        Semilla para reproducibilidad.
    include_originals : bool
        Si True, las imágenes originales también se incluyen en el JSON de salida.
    """

    ALL_AUGMENTATIONS = [
        "hflip", "vflip",
        "brightness", "contrast", "saturation", "hue", "blur",
    ]

    # Transformaciones que modifican la geometría
    GEOMETRIC = {"hflip", "vflip"}

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        augmentations: list[str] | None = None,
        augmentations_per_image: int = 3,
        max_transforms_per_sample: int = 3,
        brightness_range: tuple[float, float] = (0.6, 1.4),
        contrast_range: tuple[float, float] = (0.6, 1.4),
        saturation_range: tuple[float, float] = (0.5, 1.5),
        hue_range: tuple[float, float] = (-30.0, 30.0),
        blur_radius_range: tuple[float, float] = (0.5, 2.0),
        seed: int | None = 42,
        include_originals: bool = True,
    ) -> None:
        self.images_dir = Path(input_dir)
        self.annotations_path = self.images_dir / "_annotations.coco.json"
        self.output_dir = Path(output_dir)
        self.augmentations = augmentations if augmentations is not None else self.ALL_AUGMENTATIONS
        self.augmentations_per_image = augmentations_per_image
        self.max_transforms_per_sample = max_transforms_per_sample
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.blur_radius_range = blur_radius_range
        self.seed = seed
        self.include_originals = include_originals
        self.logger = get_logger(__name__, level=logging.DEBUG)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._validate_augmentations()

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """
        Ejecuta el proceso completo de aumentación.

        Devuelve la ruta al nuevo annotations.json generado.
        """
        out_images_dir = self.output_dir
        out_images_dir.mkdir(parents=True, exist_ok=True)

        with open(self.annotations_path, "r") as f:
            coco: dict[str, Any] = json.load(f)

        self._validate_coco(coco)

        # Índices de utilidad
        id_to_image = {img["id"]: img for img in coco["images"]}
        anns_by_image: dict[int, list[dict]] = {img["id"]: [] for img in coco["images"]}
        for ann in coco["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        new_images: list[dict] = []
        new_annotations: list[dict] = []
        next_image_id = max(img["id"] for img in coco["images"]) + 1
        next_ann_id = (max((a["id"] for a in coco["annotations"]), default=0)) + 1

        if self.include_originals:
            # Copiar imágenes originales y sus anotaciones sin cambios
            for img_info in coco["images"]:
                src = self.images_dir / img_info["file_name"]
                dst = out_images_dir / img_info["file_name"]
                if src.exists() and not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    Image.open(src).save(dst)
            new_images.extend(copy.deepcopy(coco["images"]))
            new_annotations.extend(copy.deepcopy(coco["annotations"]))

        total_generated = 0

        for img_info in coco["images"]:
            img_path = self.images_dir / img_info["file_name"]
            if not img_path.exists():
                self.logger.warning(f"Imagen no encontrada, se omite: {img_path}")
                continue

            image = Image.open(img_path).convert("RGB")
            annotations = anns_by_image.get(img_info["id"], [])

            for aug_idx in range(self.augmentations_per_image):
                # Seleccionar un subconjunto aleatorio de transformaciones
                n_transforms = random.randint(1, self.max_transforms_per_sample)
                chosen = random.sample(
                    self.augmentations,
                    min(n_transforms, len(self.augmentations)),
                )

                aug_image, aug_anns, new_w, new_h = self._apply_pipeline(
                    image, annotations, chosen,
                    img_info["width"], img_info["height"],
                )

                # Nombre de archivo único
                stem = Path(img_info["file_name"]).stem
                ext = Path(img_info["file_name"]).suffix or ".jpg"
                aug_filename = f"{stem}_aug{aug_idx:04d}{ext}"
                aug_path = out_images_dir / aug_filename
                aug_image.save(aug_path)

                # Registro en COCO
                new_img_entry = {
                    "id": next_image_id,
                    "file_name": aug_filename,
                    "width": new_w,
                    "height": new_h,
                }
                new_images.append(new_img_entry)

                for ann in aug_anns:
                    new_ann = copy.deepcopy(ann)
                    new_ann["id"] = next_ann_id
                    new_ann["image_id"] = next_image_id
                    new_annotations.append(new_ann)
                    next_ann_id += 1

                next_image_id += 1
                total_generated += 1

        # Construir JSON de salida
        out_coco: dict[str, Any] = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco.get("categories", []),
            "images": new_images,
            "annotations": new_annotations,
        }

        out_json_path = self.output_dir / "_annotations.coco.json"
        with open(out_json_path, "w") as f:
            json.dump(out_coco, f, indent=2)

        self.logger.debug(f"{total_generated} imágenes aumentadas generadas en '{self.output_dir}'.")
        self.logger.debug(f"Anotaciones guardadas en '{out_json_path}'.")
        return out_json_path

    # ------------------------------------------------------------------
    # Pipeline de transformación
    # ------------------------------------------------------------------

    def _apply_pipeline(
        self,
        image: Image.Image,
        annotations: list[dict],
        transforms: list[str],
        width: int,
        height: int,
    ) -> tuple[Image.Image, list[dict], int, int]:
        """Aplica una secuencia de transformaciones a imagen y anotaciones."""
        img = image.copy()
        anns = copy.deepcopy(annotations)
        w, h = width, height

        for t in transforms:
            img, anns, w, h = self._apply_single(img, anns, t, w, h)

        return img, anns, w, h

    def _apply_single(
        self,
        image: Image.Image,
        annotations: list[dict],
        transform: str,
        width: int,
        height: int,
    ) -> tuple[Image.Image, list[dict], int, int]:
        """Despacha y aplica una única transformación."""
        if transform == "hflip":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            for ann in annotations:
                ann["segmentation"] = _flip_polygons_horizontal(ann["segmentation"], width)
                ann["bbox"] = _recompute_bbox(ann["segmentation"])
                ann["area"] = _recompute_area(ann["segmentation"])

        elif transform == "vflip":
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            for ann in annotations:
                ann["segmentation"] = _flip_polygons_vertical(ann["segmentation"], height)
                ann["bbox"] = _recompute_bbox(ann["segmentation"])
                ann["area"] = _recompute_area(ann["segmentation"])

        elif transform in ("rot90", "rot180"):
            times = 1 if transform == "rot90" else 2
            # PIL ROTATE_90 rota 90° antihorario
            pil_ops = {1: Image.ROTATE_90, 2: Image.ROTATE_180}
            image = image.transpose(pil_ops[times])
            for ann in annotations:
                new_segs, width, height = _rotate_polygons_90(
                    ann["segmentation"], width, height, times
                )
                ann["segmentation"] = new_segs
                ann["bbox"] = _recompute_bbox(ann["segmentation"])
                ann["area"] = _recompute_area(ann["segmentation"])

        elif transform == "brightness":
            factor = random.uniform(*self.brightness_range)
            image = ImageEnhance.Brightness(image).enhance(factor)

        elif transform == "contrast":
            factor = random.uniform(*self.contrast_range)
            image = ImageEnhance.Contrast(image).enhance(factor)

        elif transform == "saturation":
            factor = random.uniform(*self.saturation_range)
            image = ImageEnhance.Color(image).enhance(factor)

        elif transform == "hue":
            image = self._shift_hue(image, random.uniform(*self.hue_range))

        elif transform == "blur":
            radius = random.uniform(*self.blur_radius_range)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        return image, annotations, width, height

    # ------------------------------------------------------------------
    # Tono (hue shift) — implementado manualmente vía HSV
    # ------------------------------------------------------------------

    @staticmethod
    def _shift_hue(image: Image.Image, degrees: float) -> Image.Image:
        """Rota el canal H de la imagen HSV en 'degrees' grados."""
        arr = np.array(image, dtype=np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

        max_c = np.max(arr, axis=2)
        min_c = np.min(arr, axis=2)
        delta = max_c - min_c

        epsilon = 0.0001
        # Saturación y valor
        s = np.where(max_c > 0, delta / np.maximum(max_c, epsilon), 0.0)
        v = max_c

        # Tono
        h = np.zeros_like(max_c)
        mask_r = (max_c == r) & (delta > 0)
        mask_g = (max_c == g) & (delta > 0)
        mask_b = (max_c == b) & (delta > 0)
        h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360
        h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120
        h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240

        h = (h + degrees) % 360  # desplazamiento

        # HSV → RGB
        hi = (h / 60).astype(int) % 6
        f = h / 60 - np.floor(h / 60)
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        out = np.zeros_like(arr)
        for val, (rv, gv, bv) in enumerate([(v, t, p), (q, v, p), (p, v, t),
                                             (p, q, v), (t, p, v), (v, p, q)]):
            mask = hi == val
            out[mask, 0] = rv[mask]
            out[mask, 1] = gv[mask]
            out[mask, 2] = bv[mask]

        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    # ------------------------------------------------------------------
    # Validaciones
    # ------------------------------------------------------------------

    def _validate_augmentations(self) -> None:
        unknown = set(self.augmentations) - set(self.ALL_AUGMENTATIONS)
        if unknown:
            raise ValueError(
                f"Aumentaciones desconocidas: {unknown}. "
                f"Opciones válidas: {self.ALL_AUGMENTATIONS}"
            )

    @staticmethod
    def _validate_coco(coco: dict) -> None:
        for key in ("images", "annotations", "categories"):
            if key not in coco:
                raise ValueError(f"El JSON COCO no contiene la clave '{key}'.")
        for ann in coco["annotations"]:
            if not isinstance(ann.get("segmentation"), list):
                raise ValueError(
                    f"La anotación id={ann.get('id')} no tiene segmentación de tipo lista."
                )
            for seg in ann["segmentation"]:
                if not isinstance(seg, list) or len(seg) % 2 != 0:
                    raise ValueError(
                        f"Segmento inválido en anotación id={ann.get('id')}: "
                        "debe ser una lista plana de coordenadas x,y."
                    )


# ---------------------------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    aug = Augmenter(
        images_dir="data/images",
        annotations_path="data/annotations.json",
        output_dir="data/augmented",
        # Aumentaciones a usar (todas por defecto)
        augmentations=[
            "hflip", "vflip", "rot90", "rot180",
            "brightness", "contrast", "saturation", "hue", "blur",
        ],
        # 4 versiones aumentadas por imagen original
        augmentations_per_image=4,
        # Cada versión combinará hasta 3 transformaciones distintas
        max_transforms_per_sample=3,
        # Rangos de intensidad para las transformaciones de color
        brightness_range=(0.6, 1.4),
        contrast_range=(0.6, 1.4),
        saturation_range=(0.5, 1.5),
        hue_range=(-30.0, 30.0),
        blur_radius_range=(0.5, 2.0),
        seed=42,
        include_originals=True,
    )
    aug.run()