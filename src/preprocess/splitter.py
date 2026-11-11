import math
import shutil
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from ..core.params import get_global_config, get_preprocess_config
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


def _assign_by_deficit(
    image_ids: list[int],
    labels: np.ndarray,
    frozen: dict[int, str],
    ratios: dict[str, float],
) -> dict[int, str]:
    """Place new images where the split is furthest below its target, label-wise first.

    Existing assignments are kept; new images are handled rarest labels first so a single new
    image (where the multilabel stratifier cannot run) still goes where it is most needed.
    """
    splits = list(ratios)
    indices = {image_id: idx for idx, image_id in enumerate(image_ids)}
    assigned = dict(frozen)
    image_counts = {split: 0 for split in splits}
    label_counts = {split: np.zeros(labels.shape[1], dtype=np.int64) for split in splits}
    for image_id, split in assigned.items():
        image_counts[split] += 1
        label_counts[split] += labels[indices[image_id]]
    label_totals = labels.sum(axis=0)
    pending = [image_id for image_id in image_ids if image_id not in assigned]

    def rarity(image_id):
        present = labels[indices[image_id]].astype(bool)
        return (int(label_totals[present].min()) if present.any() else len(image_ids), indices[image_id])

    seen_labels = sum(label_counts.values())
    for image_id in sorted(pending, key=rarity):
        row = labels[indices[image_id]]
        seen_labels = seen_labels + row
        total_images = sum(image_counts.values()) + 1
        present = row.astype(bool)
        scores = {
            split: (
                float((ratios[split] * seen_labels[present] - label_counts[split][present]).sum()),
                ratios[split] * total_images - image_counts[split],
                ratios[split],
                -index,
            )
            for index, split in enumerate(splits)
        }
        split = max(splits, key=scores.__getitem__)
        assigned[image_id] = split
        image_counts[split] += 1
        label_counts[split] += row
    return assigned


def assign_splits(
    coco: dict, ratios: tuple[float, float, float], seed: int, frozen: dict[int, str] | None = None,
) -> dict[int, str]:
    """Split assignment per COCO image id; `ratios` is (train, valid, test).

    Without previous assignments the whole dataset is stratified as before; otherwise the
    images in `frozen` keep their split and only the remaining ones are distributed.
    """
    validate_ratios(*ratios)
    validate_coco(coco)
    image_ids, labels = _build_label_matrix(coco)
    known = set(image_ids)
    frozen = {image_id: split for image_id, split in (frozen or {}).items() if image_id in known}
    if not frozen:
        split_ids = _stratify(image_ids, labels, ratios[0], ratios[1], seed)
        return {image_id: split for split, ids in split_ids.items() for image_id in ids}
    return _assign_by_deficit(image_ids, labels, frozen, dict(zip(("train", "valid", "test"), ratios, strict=True)))


def split_annotations(coco: dict) -> dict[str, dict]:
    global_cfg = get_global_config()
    cfg = get_preprocess_config()
    validate_ratios(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)
    validate_coco(coco)
    image_ids, labels = _build_label_matrix(coco)
    split_ids = _stratify(
        image_ids, labels, cfg.train_ratio, cfg.val_ratio, global_cfg.seed
    )
    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = group_annotations_by_image(coco)
    return {
        split: {
            **coco,
            "images": [images_by_id[image_id] for image_id in ids],
            "annotations": [
                ann for image_id in ids for ann in anns_by_image.get(image_id, [])
            ],
        }
        for split, ids in split_ids.items()
    }


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
    if image_action not in ("copy", "move", "ignore"):
        raise ValueError("image_action debe ser 'copy', 'move' o 'ignore'.")
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir) if output_dir is not None else dataset_dir
    annotations_path = dataset_dir / "_annotations.coco.json"
    coco = read_annotations(annotations_path)
    split_ids = {
        split: [img["id"] for img in subset["images"]]
        for split, subset in split_annotations(coco).items()
    }
    output_paths = _write_splits(coco, split_ids, dataset_dir, output_dir, image_action)
    annotations_path.unlink()
    return output_paths
