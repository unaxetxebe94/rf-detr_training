import os
import re
import cv2
import json
import random
import logging
import numpy as np
import pyvips
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box
from shapely import affinity
from shapely.strtree import STRtree
from shapely.geometry.base import BaseGeometry
from dataclasses import dataclass
from logger import get_logger
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass(frozen=True)
class _ROI:
    x: int
    y: int
    w: int
    h: int


class TileCreator:
    """
    Loads images, optionally crops to a bright ROI and/or resizes them
    in memory, then slices the result into tiles and writes only the tiles
    to disk together with their COCO annotations.

    All intermediate transforms (crop + resize) happen in memory so no
    temporary files are written between stages.

    Args:
        in_dir_path         : Folder that contains the source images and
                              _annotations.coco.json.
        out_dir_path        : Destination folder for tiles and their JSON.
        tile_size           : Side length (px) of each tile in the
                              *post-transform* coordinate space.
        black_threshold     : Tiles whose average pixel value is below
                              this value are discarded (dark-border guard).
        saving_prob         : Probability [0, 1] of saving a tile that
                              contains NO annotations. 0 = never save
                              empty tiles; 1 = always save them.
        n_jobs              : Number of parallel worker threads.
        resize_factor       : Scale factor applied to width and height
                              (e.g. 0.5 halves each dimension).
                              1.0 means no resize.
        crop                : If True, detect the bright central ROI in
                              each image and crop to it before any resize.
        roi_threshold       : Pixel brightness (0-255) used to separate
                              the dark background from the ROI. Pixels
                              above this value are treated as foreground.
        roi_padding         : Extra pixels added around the detected ROI
                              bounding box (clamped to image bounds).
    """

    def __init__(
        self,
        in_dir_path: str,
        out_dir_path: str,
        tile_size: int,
        black_threshold: int = 10,
        saving_prob: float = 0.0,
        n_jobs: int = 1,
        resize_factor: float = 1.0,
        crop: bool = False,
        roi_threshold: int = 20,
        roi_padding: int = 0,
    ):
        self.in_dir_path = in_dir_path
        self.out_dir_path = out_dir_path
        self.tile_size = tile_size
        self.black_threshold = black_threshold
        self.saving_prob = saving_prob
        self.n_jobs = n_jobs
        self.resize_factor = resize_factor
        self.crop = crop
        self.roi_threshold = roi_threshold
        self.roi_padding = roi_padding
        self.logger = get_logger(__name__, level=logging.DEBUG)

    # ------------------------------------------------------------------ #
    # ROI detection (operates on a small thumbnail – fast for huge TIFFs)
    # ------------------------------------------------------------------ #

    def _detect_roi(self, img_path: str, orig_w: int, orig_h: int):
        THUMB_SIZE = 1024
        thumb = pyvips.Image.thumbnail(img_path, THUMB_SIZE)
        scale_x = orig_w / thumb.width
        scale_y = orig_h / thumb.height

        if thumb.bands >= 3:
            blue = thumb[2]
        else:
            blue = thumb  # fallback si ya es monocanal

        arr = np.ndarray(
            buffer=blue.write_to_memory(),
            dtype=np.uint8,
            shape=(blue.height, blue.width),
        )

        # 🔥 Threshold automático (Otsu)
        _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = thresh > 0

        # Debug útil
        self.logger.debug(f"{img_path}: porcentaje máscara = {mask.mean():.3f}")

        rows_any = mask.any(axis=1)
        cols_any = mask.any(axis=0)

        if not rows_any.any():
            self.logger.warning(
                f"{img_path}: no ROI found. Falling back to full image."
            )
            return _ROI(0, 0, orig_w, orig_h)

        row_min = int(np.argmax(rows_any))
        row_max = int(len(rows_any) - 1 - np.argmax(rows_any[::-1]))
        col_min = int(np.argmax(cols_any))
        col_max = int(len(cols_any) - 1 - np.argmax(cols_any[::-1]))

        x  = max(0,      int(col_min * scale_x) - self.roi_padding)
        y  = max(0,      int(row_min * scale_y) - self.roi_padding)
        x2 = min(orig_w, int((col_max + 1) * scale_x) + self.roi_padding)
        y2 = min(orig_h, int((row_max + 1) * scale_y) + self.roi_padding)

        roi = _ROI(x=x, y=y, w=x2 - x, h=y2 - y)
        self.logger.debug(f"{img_path}: detected ROI -> {roi}")
        return roi

    # ------------------------------------------------------------------ #
    # Annotation coordinate transform (crop offset + resize scale)
    # ------------------------------------------------------------------ #

    def _transform_annotations(
        self,
        img_anns: list[dict],
        roi: _ROI | None,
        f: float,
    ) -> list[dict]:
        """
        Return a new list of annotations whose coordinates have been
        shifted by the ROI offset and scaled by f.

        Annotations that fall completely outside the ROI are discarded.
        This mirrors exactly what the image pipeline does so that the
        tiling geometry index always works in post-transform space.
        """
        off_x = roi.x if roi else 0
        off_y = roi.y if roi else 0
        roi_w  = roi.w if roi else float("inf")
        roi_h  = roi.h if roi else float("inf")

        transformed = []
        for ann in img_anns:
            new_ann = dict(ann)

            # ---- bbox ------------------------------------------------
            if "bbox" in ann:
                x, y, w, h = ann["bbox"]
                x2, y2 = x + w, y + h

                if roi is not None:
                    x -= off_x; x2 -= off_x
                    y -= off_y; y2 -= off_y
                    x  = max(0.0, x);   x2 = min(float(roi_w), x2)
                    y  = max(0.0, y);   y2 = min(float(roi_h), y2)
                    if x2 <= x or y2 <= y:
                        continue  # completely outside the crop region

                new_ann["bbox"] = [x * f, y * f, (x2 - x) * f, (y2 - y) * f]
                new_ann["area"] = new_ann["bbox"][2] * new_ann["bbox"][3]

            # ---- polygon segmentation --------------------------------
            if "segmentation" in ann and isinstance(ann["segmentation"], list):
                adjusted = []
                for poly in ann["segmentation"]:
                    new_poly = []
                    for i, coord in enumerate(poly):
                        if i % 2 == 0:  # x
                            new_poly.append(max(0.0, coord - off_x) * f)
                        else:           # y
                            new_poly.append(max(0.0, coord - off_y) * f)
                    adjusted.append(new_poly)
                new_ann["segmentation"] = adjusted

            transformed.append(new_ann)

        return transformed

    # ------------------------------------------------------------------ #
    # Geometry helpers (unchanged from original TileCreator)
    # ------------------------------------------------------------------ #

    def _segmentation_to_polygons(self, segmentation: list) -> list[Polygon]:
        if not segmentation:
            return []
        polys_list = (
            segmentation if isinstance(segmentation[0], list) else [segmentation]
        )
        result = []
        for seg in polys_list:
            if len(seg) < 6:
                continue
            try:
                poly = Polygon(np.array(seg, dtype=np.float64).reshape(-1, 2))
                if not poly.is_empty and poly.area > 0:
                    result.append(poly)
            except Exception:
                continue
        return result

    def _shapely_geom_to_coco_segs(self, geom: BaseGeometry) -> list:
        if geom.is_empty:
            return []
        if geom.geom_type == "Polygon":
            coords = np.array(geom.exterior.coords).ravel().tolist()
            return [coords] if len(coords) >= 6 else []
        if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
            segs = []
            for part in geom.geoms:
                segs.extend(self._shapely_geom_to_coco_segs(part))
            return segs
        return []

    def _segmentation_to_bbox(self, segs: list) -> list[float]:
        if not segs:
            return [0.0, 0.0, 0.0, 0.0]
        arr = np.array([coord for seg in segs for coord in seg])
        x, y = arr[0::2], arr[1::2]
        return [float(x.min()), float(y.min()), float(np.ptp(x)), float(np.ptp(y))]

    # ------------------------------------------------------------------ #
    # Spatial index
    # ------------------------------------------------------------------ #

    def _build_geometry_index(
        self, img_anns: list[dict]
    ) -> tuple[list, list, STRtree | None]:
        geometries, ann_map = [], []
        for ann in img_anns:
            polys = self._segmentation_to_polygons(ann.get("segmentation"))
            if not polys:
                bx, by, bw, bh = ann.get("bbox", [0, 0, 0, 0])
                polys = [shapely_box(bx, by, bx + bw, by + bh)]
            for poly in polys:
                if poly.is_valid and not poly.is_empty:
                    geometries.append(poly)
                    ann_map.append(ann)
        tree = STRtree(geometries) if geometries else None
        return geometries, ann_map, tree

    # ------------------------------------------------------------------ #
    # Per-tile annotation intersection
    # ------------------------------------------------------------------ #

    def _calculate_overlap(
        self,
        tile_x: int,
        tile_y: int,
        tile_w: int,
        tile_h: int,
        curr_img_id: int,
        curr_ann_id: int,
        geometries: list,
        ann_map: list,
        tree: STRtree,
    ) -> tuple[list[dict], int]:
        tile_box = shapely_box(tile_x, tile_y, tile_x + tile_w, tile_y + tile_h)
        new_anns = []

        for idx in tree.query(tile_box):
            geom = geometries[idx]
            try:
                intersection = geom.intersection(tile_box)
                if intersection.is_empty:
                    continue
                intersection = affinity.translate(
                    intersection, xoff=-tile_x, yoff=-tile_y
                )
                coco_segs = self._shapely_geom_to_coco_segs(intersection)
                if not coco_segs:
                    continue

                bbox = self._segmentation_to_bbox(coco_segs)
                area = float(bbox[2] * bbox[3])
                original_ann = ann_map[idx]

                new_anns.append(
                    {
                        "id": curr_ann_id,
                        "image_id": curr_img_id,
                        "category_id": original_ann.get("category_id"),
                        "segmentation": coco_segs,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": original_ann.get("iscrowd", 0),
                    }
                )
                curr_ann_id += 1
            except Exception as e:
                self.logger.warning(
                    f"Intersection error at tile ({tile_x},{tile_y}): {e}"
                )

        return new_anns, curr_ann_id

    # ------------------------------------------------------------------ #
    # Core per-image processing
    # ------------------------------------------------------------------ #

    def process_image(
        self,
        img_info: dict,
        anns: dict,
        start_img_id: int,
        start_ann_id: int,
    ) -> tuple[list, list, int, int]:
        img_name = img_info["file_name"]
        img_path = os.path.join(self.in_dir_path, img_name)

        if not os.path.exists(img_path):
            self.logger.warning(f"Imagen no encontrada: {img_path}")
            return [], [], start_img_id, start_ann_id

        # ── 1. Load (random access for efficient regional decoding) ──────
        try:
            img = pyvips.Image.new_from_file(img_path, access="random")
        except Exception as e:
            self.logger.warning(f"Cannot open {img_path}: {e}")
            return [], [], start_img_id, start_ann_id

        orig_w, orig_h = img.width, img.height
        base_name = os.path.splitext(img_name)[0]

        # ── 2. Optional ROI crop (in memory) ────────────────────────────
        roi: _ROI | None = None
        if self.crop:
            roi = self._detect_roi(img_path, orig_w, orig_h)
            img = img.crop(roi.x, roi.y, roi.w, roi.h)

        # ── 3. Optional resize (in memory) ──────────────────────────────
        f = self.resize_factor
        if f != 1.0:
            img = img.resize(f)

        img_w, img_h = img.width, img.height

        # ── 4. Transform annotations into the post-crop/resize space ────
        img_id = img_info.get("id")
        raw_anns = [a for a in anns.get("annotations", []) if a.get("image_id") == img_id]
        transformed_anns = self._transform_annotations(raw_anns, roi, f)

        if not transformed_anns:
            self.logger.info(f"Sin anotaciones para {img_name}, se omite.")
            return [], [], start_img_id, start_ann_id

        # ── 5. Build spatial index (once per image) ──────────────────────
        geometries, ann_map, tree = self._build_geometry_index(transformed_anns)
        if not tree:
            return [], [], start_img_id, start_ann_id

        new_images, new_annotations = [], []
        curr_img_id = start_img_id
        curr_ann_id = start_ann_id

        # ── 6. Tile ──────────────────────────────────────────────────────
        for tile_y in range(0, img_h, self.tile_size):
            for tile_x in range(0, img_w, self.tile_size):
                tile_w = min(self.tile_size, img_w - tile_x)
                tile_h = min(self.tile_size, img_h - tile_y)

                # Compute annotations before any I/O
                tile_anns, next_ann_id = self._calculate_overlap(
                    tile_x, tile_y, tile_w, tile_h,
                    curr_img_id, curr_ann_id,
                    geometries, ann_map, tree,
                )

                # Skip empty tiles according to saving_prob
                if not tile_anns and self.saving_prob < random.random():
                    continue

                # Extract tile region from the already-transformed in-memory image
                final_tile_name = f"{base_name}_tile_{tile_x}_{tile_y}.png"
                final_tile_path = os.path.join(self.out_dir_path, final_tile_name)

                try:
                    tile_img = img.crop(tile_x, tile_y, tile_w, tile_h)

                    # Discard near-black tiles (border artefacts, etc.)
                    if tile_img.avg() < self.black_threshold:
                        continue

                    tile_img.write_to_file(final_tile_path)
                except Exception as e:
                    self.logger.warning(
                        f"Error al extraer tile ({tile_x},{tile_y}) de {img_name}: {e}"
                    )
                    continue

                new_images.append(
                    {
                        "id": curr_img_id,
                        "file_name": final_tile_name,
                        "width": tile_w,
                        "height": tile_h,
                    }
                )
                new_annotations.extend(tile_anns)
                curr_ann_id = next_ann_id
                curr_img_id += 1

        return new_images, new_annotations, curr_img_id, curr_ann_id

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        with open(Path(self.in_dir_path) / "_annotations.coco.json", mode="r") as f:
            anns = json.load(f)

        images_info = anns.get("images", [])
        if not images_info:
            valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            for idx, fname in enumerate(os.listdir(self.in_dir_path)):
                if Path(fname).suffix.lower() in valid_exts:
                    images_info.append({"id": idx, "file_name": fname})

        os.makedirs(self.out_dir_path, exist_ok=True)

        final_images: list[dict] = []
        final_annotations: list[dict] = []
        global_img_id = 1
        global_ann_id = 1

        self.logger.info(
            f"Iniciando procesado de {len(images_info)} imágenes "
            f"con {self.n_jobs} hilos "
            f"(resize={self.resize_factor}, crop={self.crop})..."
        )

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(self.process_image, img_info, anns, 1, 1)
                for img_info in images_info
            ]
            for future in as_completed(futures):
                try:
                    new_imgs, new_anns, _, _ = future.result()

                    local_img_map: dict[int, int] = {}
                    for img in new_imgs:
                        local_img_map[img["id"]] = global_img_id
                        img["id"] = global_img_id
                        final_images.append(img)
                        global_img_id += 1

                    for ann in new_anns:
                        ann["id"] = global_ann_id
                        ann["image_id"] = local_img_map[ann["image_id"]]
                        final_annotations.append(ann)
                        global_ann_id += 1

                except Exception as e:
                    self.logger.error(f"Error procesando imagen en worker: {e}")

        self.logger.info("Procesado completado.")

        final_dict = {
            "images": final_images,
            "annotations": final_annotations,
            "categories": anns.get("categories", []),
        }

        with open(Path(self.out_dir_path) / "_annotations.coco.json", mode="w") as f:
            json.dump(final_dict, f)