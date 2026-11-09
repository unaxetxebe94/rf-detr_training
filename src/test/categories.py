"""Alignment between checkpoint, training, tile and original-image categories."""

from math import isclose
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from ..preprocess.annotations import group_annotations_by_image, read_annotations


def validate_checkpoint_categories(model, training_categories):
    checkpoint_names = getattr(getattr(model, "model", None), "class_names", None)
    if isinstance(checkpoint_names, (list, tuple)) and checkpoint_names:
        training_names = [
            cat["name"] for cat in training_categories
            if cat.get("supercategory") != "none"
        ]
        if [name.lower() for name in checkpoint_names] != [name.lower() for name in training_names]:
            raise ValueError(
                "Checkpoint classes/order do not match training annotations. "
                "Use --train-annotations with the original training COCO JSON. "
                f"Checkpoint: {list(checkpoint_names)}; dataset: {training_names}"
            )


def align_categories(coco, training_categories):
    training_ids = {cat["name"].lower(): cat["id"] for cat in training_categories}
    if len(training_ids) != len(training_categories):
        raise ValueError("Training category names must be unique for alignment.")
    mapping = {
        cat["id"]: training_ids[cat["name"].lower()]
        for cat in coco["categories"] if cat["name"].lower() in training_ids
    }

    def align_annotations(annotations):
        return [
            {**ann, "category_id": mapping[ann["category_id"]]}
            for ann in annotations if ann["category_id"] in mapping
        ]

    aligned = {
        **coco,
        "categories": training_categories,
        "annotations": align_annotations(coco["annotations"]),
    }
    if "source_images" in coco:
        aligned["source_images"] = []
        for source in coco["source_images"]:
            annotations = align_annotations(source["annotations"])
            aligned["source_images"].append({
                **source, "annotations": annotations,
                "has_defects": bool(annotations),
            })
    removed = [cat["name"] for cat in coco["categories"] if cat["id"] not in mapping]
    print(f"Category alignment to training IDs: {mapping}; removed classes: {removed}")
    return aligned


def _index_original_images(original_data_dirs):
    originals_by_source, originals_by_name = {}, {}
    for directory in dict.fromkeys(Path(path) for path in original_data_dirs):
        original = read_annotations(directory / "_annotations.coco.json")
        categories = {cat["id"]: cat["name"].lower() for cat in original["categories"]}
        annotations = group_annotations_by_image(original)
        for image in original["images"]:
            source_id = uuid5(
                NAMESPACE_URL, str((directory / image["file_name"]).resolve())
            ).hex
            entry = (categories, image, {
                ann["id"]: ann for ann in annotations.get(image["id"], [])
            })
            originals_by_source[source_id] = entry
            originals_by_name.setdefault(image["file_name"], []).append(entry)
    return originals_by_source, originals_by_name


def _original_category_name(ann, source, original_image, original_annotations, categories):
    original_ann = original_annotations.get(ann.get("id"))
    if original_ann is not None:
        return categories[original_ann["category_id"]]

    names = set()
    if original_image.get("width") and original_image.get("height"):
        scale_x = source["width"] / original_image["width"]
        scale_y = source["height"] / original_image["height"]
        # Resize rounding can shift the scaled box by up to one pixel.
        for candidate in original_annotations.values():
            box = candidate.get("bbox")
            if box is not None and all(
                isclose(value, original_value * scale, abs_tol=1.0)
                for value, original_value, scale in zip(
                    ann["bbox"], box, (scale_x, scale_y, scale_x, scale_y), strict=True
                )
            ):
                names.add(categories[candidate["category_id"]])
    if len(names) != 1:
        raise ValueError(
            f"Cannot uniquely align annotation {ann.get('id')} for "
            f"door {source['original_file_name']} by original ID or resized box."
        )
    return names.pop()


def align_door_categories(coco, original_data_dirs):
    test_ids = {cat["name"].lower(): cat["id"] for cat in coco["categories"]}
    if len(test_ids) != len(coco["categories"]):
        raise ValueError("Test category names must be unique for door alignment.")
    originals_by_source, originals_by_name = _index_original_images(original_data_dirs)
    sources = []
    for source in coco["source_images"]:
        entry = originals_by_source.get(source["id"])
        if entry is None:
            candidates = originals_by_name.get(source["original_file_name"], [])
            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot identify original dataset for door {source['original_file_name']}."
                )
            entry = candidates[0]
        categories, original_image, original_annotations = entry
        annotations = []
        for ann in source["annotations"]:
            name = _original_category_name(ann, source, original_image, original_annotations, categories)
            if name in test_ids:
                annotations.append({**ann, "category_id": test_ids[name]})
        sources.append({
            **source, "annotations": annotations, "has_defects": bool(annotations),
        })
    return {**coco, "source_images": sources}
