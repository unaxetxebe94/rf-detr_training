import copy
import random
import shutil
from pathlib import Path

from PIL import Image

from ..disk.params import get_global_config, get_preprocess_config
from .annotations import (
    group_annotations_by_image,
    read_annotations,
    validate_coco,
    write_annotations,
)
from .image_transforms import ALL_AUGMENTATIONS, apply_transforms


def validate_augmentation_config(cfg) -> None:
    if cfg.augmentations_per_image < 0:
        raise ValueError("augmentations_per_image no puede ser negativo.")
    if cfg.max_transforms_per_sample < 1:
        raise ValueError("max_transforms_per_sample debe ser al menos 1.")


def augmented_file_name(file_name: str, index: int) -> str:
    path = Path(file_name)
    return f"{path.stem}_aug{index:04d}{path.suffix or '.jpg'}"


def augment_image(
    image_path: Path, output_dir: Path, seed_key: str, count: int, max_transforms: int,
) -> list[list]:
    """Write `count` augmented copies of one image; returns `[file_name, width, height]` per copy.

    The RNG is seeded per image so results do not depend on which other images are processed.
    """
    rng = random.Random(seed_key)
    output_dir = Path(output_dir)
    files = []
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    try:
        for aug_idx in range(count):
            chosen = rng.sample(
                ALL_AUGMENTATIONS,
                min(rng.randint(1, max_transforms), len(ALL_AUGMENTATIONS)),
            )
            aug_image = apply_transforms(image, chosen, rng)
            aug_filename = augmented_file_name(Path(image_path).name, aug_idx)
            try:
                aug_image.save(output_dir / aug_filename)
                files.append([aug_filename, aug_image.width, aug_image.height])
            finally:
                aug_image.close()
    finally:
        image.close()
    return files


def augment_dataset(input_dir: Path, output_dir: Path | None = None) -> Path:
    global_cfg = get_global_config()
    cfg = get_preprocess_config()
    validate_augmentation_config(cfg)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    coco = read_annotations(input_dir / "_annotations.coco.json")
    validate_coco(coco, require_polygons=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    anns_by_image = group_annotations_by_image(coco)
    images = copy.deepcopy(coco["images"])
    annotations = copy.deepcopy(coco["annotations"])
    next_image_id = max((img["id"] for img in images), default=0) + 1
    next_ann_id = max((ann["id"] for ann in annotations), default=0) + 1
    total_generated = 0

    for img_info in coco["images"]:
        img_path = input_dir / img_info["file_name"]
        if not img_path.exists():
            print(f"Imagen no encontrada, se omite: {img_path}")
            continue
        original_dest = output_dir / img_info["file_name"]
        if not original_dest.exists():
            original_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, original_dest)
        generated = augment_image(
            img_path, output_dir, f"{global_cfg.seed}:{img_info['file_name']}",
            cfg.augmentations_per_image, cfg.max_transforms_per_sample,
        )
        for aug_filename, width, height in generated:
            images.append({
                **img_info,
                "id": next_image_id,
                "file_name": aug_filename,
                "width": width,
                "height": height,
            })
            for ann in anns_by_image.get(img_info["id"], []):
                new_ann = copy.deepcopy(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = next_image_id
                annotations.append(new_ann)
                next_ann_id += 1
            next_image_id += 1
            total_generated += 1

    out_json = output_dir / "_annotations.coco.json"
    write_annotations(out_json, {**coco, "images": images, "annotations": annotations})
    print(f"{total_generated} imagenes aumentadas generadas en '{output_dir}'.")
    return out_json
