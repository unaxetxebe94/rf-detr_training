import os
import re
import yaml
import json
import random
import shutil
import numpy as np
import cv2
import pyvips
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box
from shapely import affinity
from shapely.strtree import STRtree
from shapely.geometry.base import BaseGeometry
import logging
from logger import get_logger
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


class TileCreator:
    def __init__(
        self,
        in_dir_path: str,
        out_dir_path: str,
        tile_size: int,
        black_threshold: int = 10,
        saving_prob=0.0,
        n_jobs: int = 1,
    ):
        self.saving_prob = saving_prob
        self.in_dir_path = in_dir_path
        self.out_dir_path = out_dir_path
        self.tile_size = tile_size
        self.black_threshold = black_threshold
        self.n_jobs = n_jobs
        self.logger = get_logger(__name__, level=logging.DEBUG)

    # ------------------------------------------------------------------ #
    # Geometry helpers
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

    def segmentation_to_bbox(self, segs: list) -> list[float]:
        if not segs:
            return [0.0, 0.0, 0.0, 0.0]

        arr = np.array([coord for seg in segs for coord in seg])
        x, y = arr[0::2], arr[1::2]
        return [float(x.min()), float(y.min()), float(np.ptp(x)), float(np.ptp(y))]

    # ------------------------------------------------------------------ #
    # Filename helpers
    # ------------------------------------------------------------------ #

    _TILE_COORD_RE = re.compile(r"_tile_(\d+)_(\d+)\.[^.]+$")

    def _parse_tile_coords_from_filename(self, filename: str) -> tuple[int | None, int | None]:
        match = self._TILE_COORD_RE.search(filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    # ------------------------------------------------------------------ #
    # Annotation helpers
    # ------------------------------------------------------------------ #

    def _build_geometry_index(
        self, img_anns: list
    ) -> tuple[list, list, STRtree | None]:
        geometries = []
        ann_map = []

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

                bbox = self.segmentation_to_bbox(coco_segs)
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
                self.logger.warning(f"Intersection error at tile ({tile_x},{tile_y}): {e}")

        return new_anns, curr_ann_id

    # ------------------------------------------------------------------ #
    # Core tile processing
    # ------------------------------------------------------------------ #

    def process_image(
        self, img_info: dict, anns: dict, start_img_id: int, start_ann_id: int
    ) -> tuple[list, list, int, int]:
        img_name = img_info["file_name"]
        img_path = os.path.join(self.in_dir_path, img_name)

        if not os.path.exists(img_path):
            return [], [], start_img_id, start_ann_id

        try:
            # access="random" es clave: permite crops eficientes sin cargar
            # la imagen entera en RAM, pyvips la decodifica por regiones bajo demanda.
            img = pyvips.Image.new_from_file(img_path, access="random")
        except Exception as e:
            self.logger.warning(f"Cannot open {img_path}: {e}")
            return [], [], start_img_id, start_ann_id

        img_w, img_h = img.width, img.height
        base_name = os.path.splitext(img_name)[0]

        # Construimos el índice espacial una sola vez por imagen
        img_id = img_info.get("id")
        img_anns = [a for a in anns.get("annotations", []) if a.get("image_id") == img_id]
        geometries, ann_map, tree = self._build_geometry_index(img_anns)

        # Si la imagen no tiene anotaciones no hay nada que guardar
        if not tree:
            self.logger.info(f"Sin anotaciones para {img_name}, se omite.")
            return [], [], start_img_id, start_ann_id

        new_images, new_annotations = [], []
        curr_img_id = start_img_id
        curr_ann_id = start_ann_id

        for tile_y in range(0, img_h, self.tile_size):
            for tile_x in range(0, img_w, self.tile_size):
                tile_w = min(self.tile_size, img_w - tile_x)
                tile_h = min(self.tile_size, img_h - tile_y)

                # ── 1. Calcular anotaciones ANTES de tocar el disco ──────────
                tile_anns, next_ann_id = self._calculate_overlap(
                    tile_x, tile_y, tile_w, tile_h,
                    curr_img_id, curr_ann_id,
                    geometries, ann_map, tree,
                )

                # Si el tile no tiene anotaciones, lo saltamos completamente con cierta probabilidad
                if not tile_anns and self.saving_prob > random.random():
                    continue

                # ── 2. Solo ahora extraemos el tile de la imagen grande ──────
                final_tile_name = f"{base_name}_tile_{tile_x}_{tile_y}.png"
                final_tile_path = os.path.join(self.out_dir_path, final_tile_name)

                try:
                    tile_img = img.crop(tile_x, tile_y, tile_w, tile_h)

                    # Descartamos tiles casi negros (artefactos de borde, etc.)
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

    def run(self) -> None:
        """
        Itera sobre todas las imágenes y ejecuta el tiling en paralelo.
        """
        with open(Path(self.in_dir_path) / "_annotations.coco.json", mode="r") as f:
            anns = json.load(f)

        images_info = anns.get("images", [])

        if not images_info:
            valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
            for idx, fname in enumerate(os.listdir(self.in_dir_path)):
                if Path(fname).suffix.lower() in valid_exts:
                    images_info.append({"id": idx, "file_name": fname})

        final_images = []
        final_annotations = []
        global_img_id = 1
        global_ann_id = 1

        self.logger.info(
            f"Iniciando procesado de {len(images_info)} imágenes con {self.n_jobs} hilos..."
        )

        os.makedirs(self.out_dir_path, exist_ok=True)

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(self.process_image, img_info, anns, 1, 1)
                for img_info in images_info
            ]

            for future in as_completed(futures):
                try:
                    new_imgs, new_anns, _, _ = future.result()

                    # Remapeamos IDs locales → globales
                    local_img_map = {}
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
                    self.logger.error(f"Error procesando una imagen en el worker: {e}")

        self.logger.info("Procesado completado.")

        final_dict = {
            "images": final_images,
            "annotations": final_annotations,
            "categories": anns.get("categories", []),
        }

        with open(Path(self.out_dir_path) / "_annotations.coco.json", mode="w") as f:
            json.dump(final_dict, f)


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    img_dir_path = Path("data", "jeld-wen-prueba")

    with open(Path("..", "..", "params.yaml"), mode="r") as f:
        params = yaml.safe_load(f)
    model_type = params["model-type"].lower()

    tile_size_mapper = {
        "nano": 384,
        "small": 512,
        "medium": 576,
        "large": 704,
    }

    tile_creator = TileCreator(
        in_dir_path=str(img_dir_path),
        out_dir_path=str(img_dir_path),
        tile_size=tile_size_mapper[model_type],
    )
    tile_creator.run()