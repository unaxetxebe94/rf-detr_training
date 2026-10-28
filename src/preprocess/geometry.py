import numpy as np
from shapely import affinity
from shapely.strtree import STRtree
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry import box as shapely_box


def segmentation_to_polygons(segmentation: list) -> list[Polygon]:
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


def shapely_geom_to_coco_segs(geom: BaseGeometry) -> list:
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        coords = np.array(geom.exterior.coords).ravel().tolist()
        return [coords] if len(coords) >= 6 else []
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        segs = []
        for part in geom.geoms:
            segs.extend(shapely_geom_to_coco_segs(part))
        return segs
    return []


def segmentation_to_bbox(segs: list) -> list[float]:
    if not segs:
        return [0.0, 0.0, 0.0, 0.0]
    arr = np.array([coord for seg in segs for coord in seg])
    x, y = arr[0::2], arr[1::2]
    return [float(x.min()), float(y.min()), float(np.ptp(x)), float(np.ptp(y))]


def retains_min_dimension_ratio(clipped_bbox: list[float], original_bbox: list[float], min_retained_dimension_ratio: float) -> bool:
    if len(clipped_bbox) < 4 or len(original_bbox) < 4:
        return False

    original_w = float(original_bbox[2])
    original_h = float(original_bbox[3])

    if original_w <= 0 or original_h <= 0:
        return False

    clipped_w = float(clipped_bbox[2])
    clipped_h = float(clipped_bbox[3])

    return (
        clipped_w / original_w >= min_retained_dimension_ratio
        and clipped_h / original_h >= min_retained_dimension_ratio
    )


def build_geometry_index(img_anns: list[dict]) -> tuple[list, list, STRtree | None]:
    geometries, ann_map = [], []
    for ann in img_anns:
        polys = segmentation_to_polygons(ann.get("segmentation"))
        if not polys:
            bx, by, bw, bh = ann.get("bbox", [0, 0, 0, 0])
            polys = [shapely_box(bx, by, bx + bw, by + bh)]
        for poly in polys:
            if poly.is_valid and not poly.is_empty:
                geometries.append(poly)
                ann_map.append(ann)
    tree = STRtree(geometries) if geometries else None
    return geometries, ann_map, tree


def calculate_overlap(
    tile_x: int,
    tile_y: int,
    tile_w: int,
    tile_h: int,
    curr_img_id: int,
    curr_ann_id: int,
    geometries: list,
    ann_map: list,
    tree: STRtree,
    min_retained_dimension_ratio: float
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
            coco_segs = shapely_geom_to_coco_segs(intersection)
            if not coco_segs:
                continue

            bbox = segmentation_to_bbox(coco_segs)
            area = float(bbox[2] * bbox[3])
            original_ann = ann_map[idx]
            original_bbox = original_ann.get("bbox")

            if original_bbox is None:
                min_x, min_y, max_x, max_y = geom.bounds
                original_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

            if not retains_min_dimension_ratio(bbox, original_bbox, min_retained_dimension_ratio):
                print(
                    "Anotación descartada por retener menos del "
                    f"{min_retained_dimension_ratio:.0%} en alguna "
                    f"dimensión: original={original_bbox}, recortada={bbox}"
                )
                continue

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
            print(
                f"Intersection error at tile ({tile_x},{tile_y}): {e}"
            )

    return new_anns, curr_ann_id
