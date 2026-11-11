"""Incremental per-source preprocessing.

Each source output directory keeps a `_preprocess_manifest.json` describing, per original image,
the input file fingerprint, its split and the files generated from it. On every run the split
JSONs are rebuilt from the raw annotations (cheap), while tiles, augmentations and full test
images are only written when the input image or the parameters that affect them changed.
"""

import copy
import gc
import hashlib
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .annotations import (
    DATASET_INFO,
    DATASET_LICENSES,
    SPLITS,
    clean_annotation_boxes,
    group_annotations_by_image,
    read_annotations,
    remap_used_categories,
    validate_coco,
    write_annotations,
)
from .augmenter import augment_image, validate_augmentation_config
from .image import open_image
from .roi import ROI
from .splitter import assign_splits
from .tile_creation import (
    _RAM_SEMAPHORE,
    TILE_SIZE_MAPPER,
    apply_layout,
    image_source_id,
    prepare_image_for_tiles,
    sample_value,
    source_image_record,
    tile_annotations,
    tile_file_name,
    tile_regions,
    write_source_image,
    write_tile,
)

# Bump when the generated files change for the same inputs; forces a full rebuild.
PIPELINE_VERSION = 1
MANIFEST_NAME = "_preprocess_manifest.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
SOURCE_IMAGES_DIR = "source_images"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotations_hash(img_info: dict, raw_anns: list[dict], categories: dict) -> str:
    """Content hash independent of annotation/category IDs (categories by lowercase name)."""
    content = {
        "size": [img_info.get("width"), img_info.get("height")],
        "annotations": [
            {
                **{key: value for key, value in ann.items() if key not in ("id", "image_id", "category_id")},
                "category": categories.get(ann.get("category_id"), str(ann.get("category_id"))),
            }
            for ann in raw_anns
        ],
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()


def _write_json_atomic(path: Path, data: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(temporary, path)


def load_manifest(output_dir: Path, input_dir: Path) -> dict | None:
    """Previous manifest, or None when it is missing, corrupt or from another pipeline/source."""
    path = Path(output_dir) / MANIFEST_NAME
    try:
        manifest = read_annotations(path)
    except (OSError, ValueError):
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != PIPELINE_VERSION
        or manifest.get("source") != str(Path(input_dir).resolve())
        or not isinstance(manifest.get("images"), dict)
    ):
        return None
    return manifest


def _unlink(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


class _ImageJob:
    """Brings the generated files of one original image up to date."""

    def __init__(self, ctx: dict, img_info: dict, raw_anns: list[dict], split: str, old: dict | None):
        self.ctx = ctx
        self.img_info = img_info
        self.raw_anns = raw_anns
        self.split = split
        self.old = old
        self.img_path = ctx["input_dir"] / img_info["file_name"]
        self.source_id = image_source_id(self.img_path)
        self._image = None
        self._prepared = None
        self.counts = Counter()

    def image(self):
        if self._image is None:
            self._image = open_image(str(self.img_path))
        return self._image

    def prepared(self, layout: dict):
        if self._prepared is None:
            roi = ROI(*layout["roi"]) if layout["roi"] else None
            self._prepared = apply_layout(self.image(), roi, layout["scale"])
        return self._prepared

    def run(self) -> dict | None:
        with _RAM_SEMAPHORE:
            try:
                return self._run()
            except Exception as error:
                print(f"Failed to open {self.img_path}: {error}")
                return None
            finally:
                self._image = self._prepared = None
                gc.collect()

    def _file_fingerprint(self) -> dict:
        stat = self.img_path.stat()
        old_file = (self.old or {}).get("file") or {}
        if old_file.get("size") == stat.st_size and old_file.get("mtime_ns") == stat.st_mtime_ns:
            return old_file
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": _file_sha256(self.img_path)}

    def _layout(self, file_info: dict) -> tuple[dict, bool]:
        ctx = self.ctx
        key = {
            "sha256": file_info["sha256"], "tile_size": ctx["tile_size"], "resize": ctx["cfg"].resize,
            "apply_roi": ctx["cfg"].apply_roi, "version": PIPELINE_VERSION,
        }
        old_layout = (self.old or {}).get("layout")
        if old_layout and old_layout.get("key") == key:
            return old_layout, True
        prepared, roi, scale = prepare_image_for_tiles(self.image(), self.img_path, ctx["tile_size"], ctx["cfg"])
        self._prepared = prepared
        layout = {
            "key": key, "roi": [roi.x, roi.y, roi.w, roi.h] if roi else None, "scale": scale,
            "width": prepared.width, "height": prepared.height,
        }
        return layout, False

    def _run(self) -> dict:
        ctx, old, split = self.ctx, self.old or {}, self.split
        output_dir = ctx["output_dir"]
        file_info = self._file_fingerprint()
        layout, layout_reused = self._layout(file_info)
        roi = ROI(*layout["roi"]) if layout["roi"] else None
        regions = tile_regions(layout["width"], layout["height"], ctx["tile_size"])
        region_anns = tile_annotations(
            self.raw_anns, roi, layout["scale"], regions, ctx["cfg"].min_retained_dimension_ratio,
        )
        save_empty = split != "train" or sample_value(ctx["seed"], self.source_id) < ctx["cfg"].saving_prob
        origin_x = roi.x * layout["scale"] if roi else 0
        origin_y = roi.y * layout["scale"] if roi else 0

        selected = {}
        for (x, y, w, h), anns in zip(regions, region_anns, strict=True):
            if anns or save_empty:
                name = tile_file_name(self.img_path, self.source_id, x, y)
                selected[name] = ([x, y, w, h], anns, [x + origin_x, y + origin_y])

        old_split = old.get("split")
        old_augs = old.get("augmentations") or {}
        for name in old.get("tiles") or {}:
            old_path = output_dir / old_split / name
            new_path = output_dir / split / name
            if not layout_reused or name not in selected:
                self.counts["tiles_deleted"] += _unlink(old_path)
            elif old_split != split and old_path.is_file():
                new_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(old_path, new_path)
                self.counts["tiles_moved"] += 1

        tiles = {}
        written = set()
        for name, (region, _, _) in selected.items():
            path = output_dir / split / name
            if not (layout_reused and path.is_file()):
                write_tile(self.prepared(layout), region, path)
                written.add(name)
                self.counts["tiles_written"] += 1
            tiles[name] = region

        augmentations = self._update_augmentations(selected, written, old_augs, layout_reused)
        source_image = self._update_source_image(file_info, old)

        status = "new"
        if self.old:
            status = "reused"
            if old.get("file", {}).get("sha256") != file_info["sha256"]:
                status = "modified"
            elif old.get("annotations_hash") != ctx["annotation_hashes"][self.source_id]:
                status = "modified"
            elif old_split != split:
                status = "moved"
        self.counts[status] += 1

        entry = {
            "file_name": self.img_info["file_name"],
            "file": file_info,
            "annotations_hash": ctx["annotation_hashes"][self.source_id],
            "split": split,
            "layout": layout,
            "tiles": tiles,
            "augmentations": augmentations,
            "source_image": source_image and {key: source_image[key] for key in ("key", "width", "height")},
        }
        tile_records = [
            {
                "file_name": name, "width": region[2], "height": region[3],
                "source_image_id": self.source_id, "tile_origin": origin, "annotations": anns,
                "tile_to_source_scale": ctx["cfg"].resize / layout["scale"],
            }
            for name, (region, anns, origin) in selected.items()
        ]
        record = None
        if source_image:
            record = source_image_record(
                self.source_id, self.img_info, self.raw_anns, ctx["cfg"].resize,
                source_image["width"], source_image["height"],
            )
        return {
            "source_id": self.source_id, "entry": entry, "tiles": tile_records,
            "source_image": record, "counts": self.counts,
        }

    def _update_augmentations(self, selected, written, old_augs, layout_reused) -> dict:
        ctx, train_dir = self.ctx, self.ctx["output_dir"] / "train"
        key = {
            "seed": ctx["seed"], "count": ctx["cfg"].augmentations_per_image,
            "max_transforms": ctx["cfg"].max_transforms_per_sample, "version": PIPELINE_VERSION,
        }
        result = {}
        for name, old_aug in old_augs.items():
            keep = (
                self.split == "train" and layout_reused and name in selected and name not in written
                and old_aug.get("key") == key
                and all((train_dir / file_name).is_file() for file_name, _, _ in old_aug["files"])
            )
            if keep:
                result[name] = old_aug
            else:
                for file_name, _, _ in old_aug.get("files", []):
                    _unlink(train_dir / file_name)
        if self.split != "train" or key["count"] == 0:
            return result
        for name in selected:
            if name not in result:
                files = augment_image(
                    train_dir / name, train_dir, f"{ctx['seed']}:{name}",
                    key["count"], key["max_transforms"],
                )
                result[name] = {"key": key, "files": files}
                self.counts["augmentations_written"] += len(files)
        return result

    def _update_source_image(self, file_info: dict, old: dict) -> dict | None:
        ctx = self.ctx
        path = ctx["output_dir"] / "test" / SOURCE_IMAGES_DIR / f"{self.source_id}.png"
        old_source = old.get("source_image")
        if self.split != "test":
            if old_source:
                _unlink(path)
            return None
        key = {"sha256": file_info["sha256"], "resize": ctx["cfg"].resize, "version": PIPELINE_VERSION}
        if old_source and old_source.get("key") == key and path.is_file():
            return old_source
        width, height = write_source_image(self.image(), ctx["cfg"].resize, path)
        self.counts["source_images_written"] += 1
        return {"key": key, "width": width, "height": height}


def _delete_entry_files(output_dir: Path, entry: dict) -> None:
    split = entry.get("split")
    for name in entry.get("tiles") or {}:
        _unlink(output_dir / split / name)
    for aug in (entry.get("augmentations") or {}).values():
        for file_name, _, _ in aug.get("files", []):
            _unlink(output_dir / "train" / file_name)


def _assemble_split(coco: dict, results: list[dict], split: str, augmentations: bool) -> dict:
    images, annotations, sources = [], [], []
    tile_images = []
    for result in results:
        if result["entry"]["split"] != split:
            continue
        for tile in result["tiles"]:
            image = {key: value for key, value in tile.items() if key != "annotations"}
            image["id"] = len(images) + 1
            images.append(image)
            anns = clean_annotation_boxes(copy.deepcopy(tile["annotations"]), (tile["width"], tile["height"]))
            tile_images.append((image, anns, result["entry"]["augmentations"].get(tile["file_name"])))
            for ann in anns:
                annotations.append({**ann, "id": len(annotations) + 1, "image_id": image["id"]})
        if result["source_image"] is not None:
            sources.append(result["source_image"])
    if augmentations:
        for tile_image, anns, aug in tile_images:
            for file_name, width, height in (aug or {}).get("files", []):
                image = {**tile_image, "id": len(images) + 1, "file_name": file_name, "width": width, "height": height}
                images.append(image)
                for ann in anns:
                    annotations.append({**copy.deepcopy(ann), "id": len(annotations) + 1, "image_id": image["id"]})
    return {
        **coco,
        "info": dict(DATASET_INFO),
        "licenses": list(DATASET_LICENSES),
        "categories": copy.deepcopy(coco["categories"]),
        "images": images,
        "annotations": annotations,
        "source_images": sources,
    }


def _remove_orphans(output_dir: Path, datasets: dict[str, dict]) -> int:
    removed = 0
    for split, coco in datasets.items():
        split_dir = output_dir / split
        referenced = {image["file_name"] for image in coco["images"]}
        referenced.update(source["file_name"] for source in coco.get("source_images", []))
        for directory in (split_dir, split_dir / SOURCE_IMAGES_DIR):
            if not directory.is_dir():
                continue
            for path in directory.iterdir():
                relative = path.relative_to(split_dir).as_posix()
                if (
                    path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                    and relative not in referenced and str(Path(relative)) not in referenced
                ):
                    path.unlink()
                    removed += 1
    return removed


def process_source(input_dir: Path, output_dir: Path, cfg, global_cfg, *, full: bool = False) -> dict:
    """Bring one source output directory up to date and return per-run statistics."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    validate_augmentation_config(cfg)
    if not 0 <= cfg.saving_prob <= 1:
        raise ValueError("saving_prob debe estar en [0, 1].")
    coco = read_annotations(input_dir / "_annotations.coco.json")
    validate_coco(coco)
    file_names = [image["file_name"] for image in coco["images"]]
    if len(set(file_names)) != len(file_names):
        raise ValueError(f"Duplicate image file_name in {input_dir}.")
    source_ids = {image["id"]: image_source_id(input_dir / image["file_name"]) for image in coco["images"]}
    if len(set(source_ids.values())) != len(source_ids):
        raise ValueError(f"Duplicate images (same resolved path) in {input_dir}.")

    manifest = None if full else load_manifest(output_dir, input_dir)
    if manifest is None and output_dir.exists():
        shutil.rmtree(output_dir)
    previous = manifest["images"] if manifest else {}
    for split in SPLITS:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "test" / SOURCE_IMAGES_DIR).mkdir(exist_ok=True)

    split_key = {"seed": global_cfg.seed, "ratios": [cfg.train_ratio, cfg.val_ratio, cfg.test_ratio]}
    frozen = {}
    if manifest and manifest.get("split_key") == split_key:
        frozen = {
            image_id: previous[source_id]["split"]
            for image_id, source_id in source_ids.items() if source_id in previous
        }
    assignments = assign_splits(coco, tuple(split_key["ratios"]), global_cfg.seed, frozen)

    stats = Counter()
    current = set(source_ids.values())
    for source_id, entry in previous.items():
        if source_id not in current:
            _delete_entry_files(output_dir, entry)
            _unlink(output_dir / "test" / SOURCE_IMAGES_DIR / f"{source_id}.png")
            stats["deleted"] += 1

    categories = {cat["id"]: str(cat["name"]).lower() for cat in coco["categories"]}
    grouped = group_annotations_by_image(coco)
    ctx = {
        "input_dir": input_dir, "output_dir": output_dir, "cfg": cfg, "seed": global_cfg.seed,
        "tile_size": TILE_SIZE_MAPPER[global_cfg.model_type],
        "annotation_hashes": {
            source_ids[image["id"]]: _annotations_hash(image, grouped.get(image["id"], []), categories)
            for image in coco["images"]
        },
    }
    jobs = [
        _ImageJob(ctx, image, grouped.get(image["id"], []), assignments[image["id"]],
                  previous.get(source_ids[image["id"]]))
        for image in coco["images"]
    ]
    print(f"Actualizando {len(jobs)} imagenes en {output_dir}.")
    with ThreadPoolExecutor(max_workers=cfg.n_jobs) as executor:
        results = list(executor.map(lambda job: job.run(), jobs))
    for job, result in zip(jobs, results, strict=True):
        stats.update(job.counts)
        if result is None:
            stats["failed"] += 1
    results = [result for result in results if result is not None]

    datasets = {split: _assemble_split(coco, results, split, split == "train") for split in SPLITS}
    remap_used_categories(datasets)
    for split, dataset in datasets.items():
        write_annotations(output_dir / split / "_annotations.coco.json", dataset)
    _write_json_atomic(output_dir / MANIFEST_NAME, {
        "version": PIPELINE_VERSION,
        "source": str(input_dir.resolve()),
        "split_key": split_key,
        "images": {result["source_id"]: result["entry"] for result in results},
    })
    stats["orphans_removed"] += _remove_orphans(output_dir, datasets)
    return dict(stats)
