"""Select training classes consistently across fused COCO splits."""

from pathlib import Path

from .annotations import SPLITS, read_annotations, write_annotations


def validate_class_selection(classes):
    if classes is None:
        return None
    if not isinstance(classes, list) or not classes or any(
        not isinstance(name, str) or not name.strip() for name in classes
    ):
        raise ValueError("preprocess.classes must be null or a nonempty list of category names.")
    names = [name.lower() for name in classes]
    if len(set(names)) != len(names):
        raise ValueError("preprocess.classes contains duplicate names (case-insensitive).")
    return names


def filter_classes(dataset_path: Path, classes: list[str] | None = None) -> None:
    selected = validate_class_selection(classes)
    dataset_path = Path(dataset_path)
    datasets = {
        split: read_annotations(dataset_path / split / "_annotations.coco.json")
        for split in SPLITS
    }
    catalog = {}
    for split, coco in datasets.items():
        if len({cat["id"] for cat in coco["categories"]}) != len(coco["categories"]):
            raise ValueError(f"Duplicate category IDs in {split}.")
        for category in coco["categories"]:
            catalog.setdefault(category["name"].lower(), category)
    if not catalog:
        raise ValueError("The fused dataset contains no trainable defect categories.")
    if selected is None:
        selected = list(catalog)
    missing = set(selected) - catalog.keys()
    if missing:
        raise ValueError(
            f"Unknown preprocess.classes: {sorted(missing)}. "
            f"Available categories: {list(catalog)}."
        )
    new_ids = {name: index for index, name in enumerate(selected, 1)}
    categories = [
        {**catalog[name], "id": new_ids[name], "name": name}
        for name in selected
    ]

    def filter_annotations(annotations, mapping):
        return [
            {**ann, "category_id": mapping[ann["category_id"]]}
            for ann in annotations if ann["category_id"] in mapping
        ]

    for coco in datasets.values():
        mapping = {
            category["id"]: new_ids[category["name"].lower()]
            for category in coco["categories"] if category["name"].lower() in new_ids
        }

        coco["annotations"] = filter_annotations(coco["annotations"], mapping)
        for source in coco.get("source_images", []):
            source["annotations"] = filter_annotations(source["annotations"], mapping)
            source["has_defects"] = bool(source["annotations"])
        coco["categories"] = categories
    for split, coco in datasets.items():
        write_annotations(dataset_path / split / "_annotations.coco.json", coco)
    print(f"Training classes after fusion: {selected}")
