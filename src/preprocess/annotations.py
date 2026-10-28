import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .roi import ROI


SPLITS = ("train", "valid", "test")


def read_annotations(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_annotations(path: Path, coco: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)


def validate_coco(coco: dict, *, require_polygons: bool = False) -> None:
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"El JSON COCO no contiene la clave '{key}'.")
    if require_polygons:
        for ann in coco["annotations"]:
            segmentation = ann.get("segmentation")
            if not isinstance(segmentation, list):
                raise ValueError(
                    f"La anotacion id={ann.get('id')} no tiene segmentacion de tipo lista."
                )
            for seg in segmentation:
                if not isinstance(seg, list) or len(seg) % 2:
                    raise ValueError(f"Segmento invalido en anotacion id={ann.get('id')}.")


def group_annotations_by_image(coco: dict) -> dict[int, list[dict]]:
    grouped = defaultdict(list)
    for ann in coco["annotations"]:
        grouped[ann["image_id"]].append(ann)
    return dict(grouped)


def transform_annotations(img_anns: list[dict], roi: "ROI | None", f: float) -> list[dict]:
    off_x = roi.x if roi else 0
    off_y = roi.y if roi else 0
    roi_w  = roi.w if roi else float("inf")
    roi_h  = roi.h if roi else float("inf")

    transformed = []
    for ann in img_anns:
        new_ann = dict(ann)

        if "bbox" in ann:
            x, y, w, h = ann["bbox"]
            x2, y2 = x + w, y + h

            if roi is not None:
                x -= off_x; x2 -= off_x
                y -= off_y; y2 -= off_y
                x  = max(0.0, x);   x2 = min(float(roi_w), x2)
                y  = max(0.0, y);   y2 = min(float(roi_h), y2)
                if x2 <= x or y2 <= y:
                    continue

            new_ann["bbox"] = [x * f, y * f, (x2 - x) * f, (y2 - y) * f]
            new_ann["area"] = new_ann["bbox"][2] * new_ann["bbox"][3]

        if "segmentation" in ann and isinstance(ann["segmentation"], list):
            adjusted = []
            for poly in ann["segmentation"]:
                new_poly = []
                for i, coord in enumerate(poly):
                    if i % 2 == 0:
                        new_poly.append(max(0.0, coord - off_x) * f)
                    else:
                        new_poly.append(max(0.0, coord - off_y) * f)
                adjusted.append(new_poly)
            new_ann["segmentation"] = adjusted

        transformed.append(new_ann)

    return transformed
