from pathlib import Path

from PIL import Image

from .annotations import SPLITS, read_annotations, validate_coco, write_annotations


def _correct_split(split_dir: Path) -> None:
    annotations_path = split_dir / "_annotations.coco.json"
    coco = read_annotations(annotations_path)
    validate_coco(coco)
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

    used_ids = {ann["category_id"] for ann in annotations}
    categories = [cat for cat in coco["categories"] if cat["id"] in used_ids]
    category_mapping = {cat["id"]: idx for idx, cat in enumerate(categories)}
    for cat in categories:
        cat["id"] = category_mapping[cat["id"]]
    for ann in annotations:
        ann["category_id"] = category_mapping[ann["category_id"]]

    coco["info"] = {
        "description": "Dataset used dor the training of the B1375 proyect in Biele Digital"
    }
    coco["licenses"] = ["Only usable by Biele Group or companies collaborating with Biele Group"]
    coco["categories"] = categories
    coco["annotations"] = annotations
    write_annotations(annotations_path, coco)
    print(f"Etiquetas corregidas: {split_dir}")


def correct_labels(dataset_path: Path) -> None:
    for split in SPLITS:
        _correct_split(Path(dataset_path) / split)
