import math
import shutil
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from ..params import get_global_config, get_preprocess_config
from .annotations import (
    group_annotations_by_image,
    read_annotations,
    validate_coco,
    write_annotations,
)


def validate_ratios(train: float, valid: float, test: float) -> None:
    if not math.isclose(train + valid + test, 1.0, abs_tol=0.001):
        raise ValueError("La suma de train+valid+test debe ser 1.0.")
    if any(not 0 < ratio < 1 for ratio in (train, valid, test)):
        raise ValueError("Los ratios deben estar en (0, 1).")


def _build_label_matrix(coco: dict) -> tuple[list[int], np.ndarray]:
    image_ids = [img["id"] for img in coco["images"]]
    image_indices = {image_id: idx for idx, image_id in enumerate(image_ids)}
    category_indices = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}
    labels = np.zeros((len(image_ids), len(category_indices)), dtype=np.int8)
    for ann in coco["annotations"]:
        img_idx = image_indices.get(ann["image_id"])
        cat_idx = category_indices.get(ann["category_id"])
        if img_idx is not None and cat_idx is not None:
            labels[img_idx, cat_idx] = 1
    return image_ids, labels


def _multilabel_stratified_split(
    image_ids: list[int], labels: np.ndarray, ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=1 - ratio, random_state=seed
    )
    idx_a, idx_b = next(splitter.split(np.zeros(len(image_ids)), labels))
    ids = np.array(image_ids)
    return ids[idx_a].tolist(), ids[idx_b].tolist()


def _stratify(
    image_ids: list[int],
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    train_val_ratio = train_ratio + val_ratio
    train_val_ids, test_ids = _multilabel_stratified_split(
        image_ids, labels, train_val_ratio, seed
    )
    indices = {image_id: idx for idx, image_id in enumerate(image_ids)}
    train_val_labels = labels[[indices[image_id] for image_id in train_val_ids]]
    train_ids, val_ids = _multilabel_stratified_split(
        train_val_ids, train_val_labels, train_ratio / train_val_ratio, seed + 1
    )
    return {"train": train_ids, "valid": val_ids, "test": test_ids}


def _write_splits(
    coco: dict,
    split_ids: dict[str, list[int]],
    dataset_dir: Path,
    output_dir: Path,
    image_action: str,
) -> dict[str, Path]:
    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = group_annotations_by_image(coco)
    output_paths = {}
    for split_name, ids in split_ids.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        images = [dict(images_by_id[image_id]) for image_id in ids]
        if image_action != "ignore":
            for image in images:
                src = dataset_dir / image["file_name"]
                if not src.exists():
                    src = dataset_dir / Path(image["file_name"]).name
                dest = split_dir / src.name
                if src.exists():
                    if image_action == "move":
                        shutil.move(src, dest)
                    else:
                        shutil.copy2(src, dest)
                else:
                    print(f"Imagen no encontrada: {src}")
                image["file_name"] = src.name
        split_coco = {
            **coco,
            "images": images,
            "annotations": [
                ann for image_id in ids for ann in anns_by_image.get(image_id, [])
            ],
        }
        ann_path = split_dir / "_annotations.coco.json"
        write_annotations(ann_path, split_coco)
        output_paths[split_name] = ann_path
        print(f"{split_name}: {len(images)} imagenes.")
    return output_paths


def split_dataset(
    dataset_dir: Path, output_dir: Path | None = None, *, image_action: str = "move"
) -> dict[str, Path]:
    global_cfg = get_global_config()
    cfg = get_preprocess_config()
    validate_ratios(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)
    if image_action not in ("copy", "move", "ignore"):
        raise ValueError("image_action debe ser 'copy', 'move' o 'ignore'.")
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir) if output_dir is not None else dataset_dir
    annotations_path = dataset_dir / "_annotations.coco.json"
    coco = read_annotations(annotations_path)
    validate_coco(coco)
    image_ids, labels = _build_label_matrix(coco)
    split_ids = _stratify(
        image_ids, labels, cfg.train_ratio, cfg.val_ratio, global_cfg.seed
    )
    output_paths = _write_splits(coco, split_ids, dataset_dir, output_dir, image_action)
    annotations_path.unlink()
    return output_paths
