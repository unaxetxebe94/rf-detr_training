from pathlib import Path

from PIL import Image

from .annotations import SPLITS, read_annotations, validate_coco, write_annotations


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
