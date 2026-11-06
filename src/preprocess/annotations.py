import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

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
                x -= off_x
                x2 -= off_x
                y -= off_y
                y2 -= off_y
                x = max(0.0, x)
                x2 = min(float(roi_w), x2)
                y = max(0.0, y)
                y2 = min(float(roi_h), y2)
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


def _clean_annotations(coco: dict, split_dir: Path) -> None:
    images_by_id = {img["id"]: img for img in coco["images"]}
    dimensions = {}

    annotations = []
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        if image_id not in dimensions:
            with Image.open(split_dir / images_by_id[image_id]["file_name"]) as image:
                dimensions[image_id] = image.size
        img_width, img_height = dimensions[image_id]
        x, y, w, h = ann["bbox"]
        x = min(max(x, 0), img_width)
        y = min(max(y, 0), img_height)
        w = min(max(w, 0), img_width - x)
        h = min(max(h, 0), img_height - y)
        ann["bbox"] = [x, y, w, h]
        if w >= 4 or h >= 4:
            annotations.append(ann)

    coco["info"] = {
        "description": "Dataset used dor the training of the B1375 proyect in Biele Digital"
    }
    coco["licenses"] = ["Only usable by Biele Group or companies collaborating with Biele Group"]
    coco["annotations"] = annotations


def correct_labels(dataset_path: Path) -> None:
    datasets = {}
    for split in SPLITS:
        split_dir = Path(dataset_path) / split
        coco = read_annotations(split_dir / "_annotations.coco.json")
        validate_coco(coco)
        _clean_annotations(coco, split_dir)
        datasets[split] = coco

    all_annotations = [
        ann for coco in datasets.values()
        for ann in coco["annotations"] + [
            ann for source in coco.get("source_images", [])
            for ann in source["annotations"]
        ]
    ]
    used_ids = {ann["category_id"] for ann in all_annotations}
    categories = {
        cat["id"]: cat for coco in datasets.values() for cat in coco["categories"]
    }
    kept_ids = [old_id for old_id in categories if old_id in used_ids]
    mapping = {old_id: idx for idx, old_id in enumerate(kept_ids)}
    final_categories = [
        {**categories[old_id], "id": new_id} for old_id, new_id in mapping.items()
    ]
    for ann in all_annotations:
        ann["category_id"] = mapping[ann["category_id"]]
    for split, coco in datasets.items():
        coco["categories"] = final_categories
        split_dir = Path(dataset_path) / split
        write_annotations(split_dir / "_annotations.coco.json", coco)
        print(f"Etiquetas corregidas: {split_dir}")
