"""In-memory tile plan: the same split, tiles and labels as the disk preprocessing, without writing files.

Each source is split per original image (multilabel stratified), every original is laid out once
(ROI + resize, reading only the header and a thumbnail) and its tiles and tile annotations are kept
as coordinates. Pixels are only decoded later, by the dataset, while training.
"""

import copy
import gc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ...preprocess.annotations import SPLITS, clean_annotation_boxes, group_annotations_by_image, read_annotations, validate_coco
from ...preprocess.augmenter import validate_augmentation_config
from ...preprocess.class_filter import validate_class_selection
from ...preprocess.image import open_image
from ...preprocess.roi import ROI
from ...preprocess.splitter import assign_splits
from ...preprocess.tile_creation import (
    _RAM_SEMAPHORE,
    TILE_SIZE_MAPPER,
    image_source_id,
    prepare_image_for_tiles,
    sample_value,
    tile_annotations,
    tile_regions,
)

ANNOTATION_KEYS = ("bbox", "area", "iscrowd", "category_id")


@dataclass(frozen=True)
class Original:
    """One original image and how to rebuild the image its tiles are cropped from."""

    path: str
    source_id: str
    roi: tuple[int, int, int, int] | None
    scale: float
    width: int
    height: int


@dataclass
class SplitPlan:
    originals: list[Original] = field(default_factory=list)
    # One entry per tile: index into `originals`, region (x, y, w, h) and annotations in tile coordinates.
    tiles: list[tuple[int, tuple[int, int, int, int], list[dict]]] = field(default_factory=list)


@dataclass
class TilePlan:
    categories: list[dict]
    splits: dict[str, SplitPlan]
    tile_size: int

    def summary(self) -> dict:
        return {
            f"{split}/{key}": value
            for split, plan in self.splits.items()
            for key, value in (
                ("originals", len(plan.originals)),
                ("tiles", len(plan.tiles)),
                ("annotations", sum(len(anns) for _, _, anns in plan.tiles)),
            )
        }


def _used_category_names(coco: dict) -> list[str]:
    used = {ann["category_id"] for ann in coco["annotations"]}
    return [str(cat["name"]).lower() for cat in coco["categories"] if cat["id"] in used]


def build_categories(cocos: list[dict], classes: list[str] | None) -> list[dict]:
    """Final categories as in the disk pipeline: IDs 1..N in `classes` order or in source fusion order."""
    catalog = list(dict.fromkeys(name for coco in cocos for name in _used_category_names(coco)))
    if not catalog:
        raise ValueError("The sources contain no trainable defect categories.")
    selected = validate_class_selection(classes)
    if selected is None:
        selected = catalog
    missing = set(selected) - set(catalog)
    if missing:
        raise ValueError(f"Unknown preprocess.classes: {sorted(missing)}. Available categories: {catalog}.")
    return [{"id": index, "name": name, "supercategory": "none"} for index, name in enumerate(selected, 1)]


def _layout(img_path: Path, tile_size: int, cfg) -> Original | None:
    with _RAM_SEMAPHORE:
        try:
            img = open_image(str(img_path))
            prepared, roi, scale = prepare_image_for_tiles(img, img_path, tile_size, cfg)
            return Original(
                path=str(img_path), source_id=image_source_id(img_path),
                roi=(roi.x, roi.y, roi.w, roi.h) if roi else None, scale=scale,
                width=prepared.width, height=prepared.height,
            )
        except Exception as error:
            print(f"Failed to open {img_path}: {error}")
            return None
        finally:
            gc.collect()


def _plan_source(input_dir: Path, coco: dict, category_ids: dict[str, int], tile_size: int, cfg, seed: int, plans):
    assignments = assign_splits(coco, (cfg.train_ratio, cfg.val_ratio, cfg.test_ratio), seed)
    names = {cat["id"]: str(cat["name"]).lower() for cat in coco["categories"]}
    grouped = group_annotations_by_image(coco)
    with ThreadPoolExecutor(max_workers=cfg.n_jobs) as executor:
        layouts = list(executor.map(
            lambda image: _layout(input_dir / image["file_name"], tile_size, cfg), coco["images"],
        ))
    for image, original in zip(coco["images"], layouts, strict=True):
        if original is None:
            continue
        split = assignments[image["id"]]
        raw_anns = [
            {**ann, "category_id": category_ids[names[ann["category_id"]]]}
            for ann in grouped.get(image["id"], []) if names.get(ann["category_id"]) in category_ids
        ]
        regions = tile_regions(original.width, original.height, tile_size)
        region_anns = tile_annotations(
            raw_anns, None if original.roi is None else ROI(*original.roi), original.scale,
            regions, cfg.min_retained_dimension_ratio,
        )
        save_empty = split != "train" or sample_value(seed, original.source_id) < cfg.saving_prob
        plan = plans[split]
        origin = len(plan.originals)
        kept = 0
        for region, anns in zip(regions, region_anns, strict=True):
            anns = clean_annotation_boxes(copy.deepcopy(anns), region[2:])
            anns = [{key: ann[key] for key in ANNOTATION_KEYS if key in ann} for ann in anns]
            if anns or save_empty:
                plan.tiles.append((origin, tuple(region), anns))
                kept += 1
        if kept:
            plan.originals.append(original)


def build_tile_plan(sources: list[Path], cfg, global_cfg) -> TilePlan:
    validate_augmentation_config(cfg)
    if not 0 <= cfg.saving_prob <= 1:
        raise ValueError("saving_prob debe estar en [0, 1].")
    sources = [Path(source) for source in sources]
    cocos = []
    for source in sources:
        coco = read_annotations(source / "_annotations.coco.json")
        validate_coco(coco)
        cocos.append(coco)
    categories = build_categories(cocos, cfg.classes)
    category_ids = {cat["name"]: cat["id"] for cat in categories}
    tile_size = TILE_SIZE_MAPPER[global_cfg.model_type]
    plans = {split: SplitPlan() for split in SPLITS}
    for source, coco in zip(sources, cocos, strict=True):
        print(f"Planificando tiles en memoria de {len(coco['images'])} imagenes de {source}.")
        _plan_source(source, coco, category_ids, tile_size, cfg, global_cfg.seed, plans)
    return TilePlan(categories=categories, splits=plans, tile_size=tile_size)
