import os
import re
import yaml
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


class TileCreator:
    def __init__(
        self,
        in_dir_path: str,
        out_dir_path: str,
        tile_size: int,
        black_threshold: int = 10,
        n_jobs: int = 1,
    ):
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

    def segmentation_to_bbox(segs: list) -> list[float]:
        if not segs:
            return [0.0, 0.0, 0.0, 0.0]

        arr = np.array([coord for seg in segs for coord in seg])
        x, y = arr[0::2], arr[1::2]
        return [float(x.min()), float(y.min()), float(x.ptp()), float(y.ptp())]

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
        """Build STRtree index from COCO annotations. Returns (geometries, ann_map, tree)."""
        geometries = []
        ann_map = []  # parallel list: ann_map[i] is the annotation for geometries[i]

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
        """Compute clipped COCO annotations that overlap with a tile."""
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

    def process_image_dzsave(
        self, img_info: dict, anns: dict, start_img_id: int, start_ann_id: int
    ) -> tuple[list, list, int, int]:
        img_name = img_info["file_name"]
        img_path = os.path.join(self.in_dir_path, img_name)

        if not os.path.exists(img_path):
            return [], [], start_img_id, start_ann_id

        try:
            img = pyvips.Image.new_from_file(img_path, access="random")
        except Exception as e:
            self.logger.warning(f"Cannot open {img_path}: {e}")
            return [], [], start_img_id, start_ann_id

        # Build annotation index (no ROI offset needed)
        img_id = img_info.get("id")
        img_anns = [a for a in anns.get("annotations", []) if a.get("image_id") == img_id]
        geometries, ann_map, tree = self._build_geometry_index(img_anns)

        # Generate tiles via dzsave
        base_name = os.path.splitext(img_name)[0]
        temp_dir = os.path.join(self.out_dir_path, f"_temp_dzi_{base_name}")
        os.makedirs(temp_dir, exist_ok=True)
        dzi_base = os.path.join(temp_dir, "tiles")

        try:
            img.dzsave(
                dzi_base,
                tile_size=self.tile_size,
                suffix=".png",
                overlap=0,
                depth="one",
            )

            tiles_dir = f"{dzi_base}_files/0"
            if not os.path.exists(tiles_dir):
                self.logger.warning(f"No tiles generated for {img_name}")
                return [], [], start_img_id, start_ann_id

            new_images, new_annotations = [], []
            curr_img_id = start_img_id
            curr_ann_id = start_ann_id

            tile_files = sorted(
                f for f in os.listdir(tiles_dir) if f.endswith(".png")
            )

            for tile_file in tile_files:
                match = re.match(r"(\d+)_(\d+)\.png$", tile_file)
                if not match:
                    self.logger.warning(f"Tile {tile_file} con nombre inesperado")
                    continue

                # dzsave names tiles as col_row, so x = col * tile_size, y = row * tile_size
                tile_x = int(match.group(1)) * self.tile_size
                tile_y = int(match.group(2)) * self.tile_size
                tile_path = os.path.join(tiles_dir, tile_file)

                # Skip near-black tiles
                try:
                    tile_vips = pyvips.Image.new_from_file(tile_path)
                    if tile_vips.avg() < self.black_threshold:
                        os.remove(tile_path)
                        continue
                    tile_w, tile_h = tile_vips.width, tile_vips.height
                except Exception:
                    continue

                # Move to output before building the record
                final_tile_name = f"{base_name}_tile_{tile_x}_{tile_y}.png"
                final_tile_path = os.path.join(self.out_dir_path, final_tile_name)
                shutil.move(tile_path, final_tile_path)

                new_images.append(
                    {
                        "id": curr_img_id,
                        "file_name": final_tile_name,
                        "width": tile_w,
                        "height": tile_h,
                    }
                )

                if tree:
                    tile_anns, curr_ann_id = self._calculate_overlap(
                        tile_x, tile_y, tile_w, tile_h,
                        curr_img_id, curr_ann_id,
                        geometries, ann_map, tree,
                    )
                    new_annotations.extend(tile_anns)

                curr_img_id += 1

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return new_images, new_annotations, curr_img_id, curr_ann_id
    

# ------------------------------------------------------------------ #
# Core tile processing
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    img_dir_path = Path("data", "jeld-wen-prueba")

    # Leemos params.yaml
    with open(Path("..", "..", "params.yaml"), mode="r") as f:
        params = yaml.safe_load(f)
    model_type = params["model-type"].lower()    

    tile_size_mapper = {
        "nano": 384,
        "small": 512,
        "medium": 576,
        "large": 704
    }

    tile_creator = TileCreator(
        in_dir_path=str(img_dir_path),
        out_dir_path=str(Path(img_dir_path, "formatted")),
        tile_size=tile_size_mapper[model_type]
    )
