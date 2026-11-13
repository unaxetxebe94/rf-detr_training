"""Incremental on-disk tile cache, independent of splits, classes and augmentation.

Per source, ``<cache>/sources/<name>_<hash>/`` keeps every tile of every original (empty ones included)
plus the full-size test images, and a ``_tile_manifest.json`` with, per original: input fingerprint,
split, layout (ROI + scale) and the tiles generated from it. Pixels are only written when the input
image, the layout parameters (model tile size, resize, apply-roi) or the tile format change; split
assignment, tile annotations, class selection and negative sampling are recomputed from the raw
annotations on every run without touching any tile.
"""

import copy
import gc
import hashlib
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...preprocess.annotations import (
    clean_annotation_boxes,
    group_annotations_by_image,
    read_annotations,
    validate_coco,
)
from ...preprocess.image import open_image
from ...preprocess.incremental import _file_sha256, _unlink, _write_json_atomic
from ...preprocess.roi import ROI
from ...preprocess.splitter import assign_splits, validate_ratios
from ...preprocess.tile_creation import (
    _RAM_SEMAPHORE,
    TILE_SIZE_MAPPER,
    apply_layout,
    image_source_id,
    prepare_image_for_tiles,
    source_image_record,
    tile_annotations,
    tile_file_name,
    tile_regions,
    write_source_image,
)

# Bump when the cached files change for the same inputs; forces a full rebuild of the cache.
CACHE_VERSION = 1
MANIFEST_NAME = "_tile_manifest.json"
TILES_DIR = "tiles"
SOURCE_IMAGES_DIR = "source_images"
TILE_EXTENSIONS = {"png": ".png", "jpeg": ".jpg", "raw": ".npy"}
ANNOTATION_KEYS = ("bbox", "area", "iscrowd")


def to_rgb_uchar(img):
    """pyvips image as 3-band sRGB uchar, the only layout the tile readers have to deal with."""
    if img.interpretation not in ("srgb", "rgb") or img.format != "uchar":
        img = img.colourspace("srgb")
    if img.format != "uchar":
        img = img.cast("uchar")
    if img.bands == 1:
        img = img.bandjoin([img, img])
    elif img.bands == 2:
        img = img.extract_band(0).bandjoin([img.extract_band(0), img.extract_band(0)])
    elif img.bands > 3:
        img = img.extract_band(0, n=3)
    return img


def cache_tile_name(img_path: Path, source_id: str, x: int, y: int, tile_format: str) -> str:
    return str(Path(tile_file_name(img_path, source_id, x, y)).with_suffix(TILE_EXTENSIONS[tile_format]))


def _temporary(path: Path) -> Path:
    # Keeps the real extension last: pyvips picks the encoder from it.
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def write_tile_file(prepared, region, path: Path, tile_format: str, jpeg_quality: int = 95) -> None:
    """Crop ``region`` from a prepared (RGB uchar) pyvips image and write it atomically."""
    x, y, w, h = region
    crop = prepared.crop(x, y, w, h)
    temporary = _temporary(path)
    try:
        if tile_format == "raw":
            array = np.ndarray(buffer=crop.write_to_memory(), dtype=np.uint8, shape=(crop.height, crop.width, 3))
            with open(temporary, "wb") as f:
                np.save(f, array, allow_pickle=False)
        elif tile_format == "jpeg":
            crop.write_to_file(str(temporary), Q=jpeg_quality)
        else:
            crop.write_to_file(str(temporary))
        os.replace(temporary, path)
    except BaseException:
        _unlink(temporary)
        raise


def read_tile_array(path: Path | str) -> np.ndarray:
    """Decode a cached tile (png/jpeg/npy) as an HxWx3 uint8 RGB array. Unicode-safe on Windows."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        import cv2

        array = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if array is None:
            raise ValueError(f"No se pudo decodificar el tile {path}.")
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(f"Tile con formato inesperado {array.shape}/{array.dtype}: {path}.")
    return array


def source_cache_dir(cache_dir: Path, input_dir: Path) -> Path:
    """Cache directory of one source, stable across reorderings of ``data.sources``."""
    resolved = str(Path(input_dir).resolve())
    return Path(cache_dir) / "sources" / f"{Path(input_dir).name}_{hashlib.sha256(resolved.encode()).hexdigest()[:10]}"


@dataclass
class SourceCache:
    """Result of updating the cache of one source: what the view needs, without reading any pixel."""

    input_dir: Path
    cache_dir: Path
    coco: dict
    # One entry per original kept: split, tiles (cache-relative file, region, annotations) and test source image.
    images: list[dict] = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)


def _load_manifest(cache_dir: Path, input_dir: Path) -> dict | None:
    try:
        manifest = read_annotations(cache_dir / MANIFEST_NAME)
    except (OSError, ValueError):
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != CACHE_VERSION
        or manifest.get("source") != str(Path(input_dir).resolve())
        or not isinstance(manifest.get("images"), dict)
    ):
        return None
    return manifest


class _ImageJob:
    """Brings the cached tiles (and the test source image) of one original up to date."""

    def __init__(self, ctx: dict, img_info: dict, raw_anns: list[dict], split: str, old: dict | None):
        self.ctx = ctx
        self.img_info = img_info
        self.raw_anns = raw_anns
        self.split = split
        self.old = old or {}
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
                print(f"Failed to cache {self.img_path}: {error}")
                return None
            finally:
                self._image = self._prepared = None
                gc.collect()

    def _file_fingerprint(self) -> dict:
        stat = self.img_path.stat()
        old_file = self.old.get("file") or {}
        if old_file.get("size") == stat.st_size and old_file.get("mtime_ns") == stat.st_mtime_ns:
            return old_file
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": _file_sha256(self.img_path)}

    def _layout(self, file_info: dict) -> tuple[dict, bool]:
        ctx = self.ctx
        key = {
            "sha256": file_info["sha256"], "tile_size": ctx["tile_size"], "resize": ctx["cfg"].resize,
            "apply_roi": ctx["cfg"].apply_roi, "version": CACHE_VERSION,
        }
        old_layout = self.old.get("layout")
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
        ctx = self.ctx
        tiles_dir = ctx["cache_dir"] / TILES_DIR
        file_info = self._file_fingerprint()
        layout, layout_reused = self._layout(file_info)
        tile_key = {"layout": layout["key"], "format": ctx["tile_format"]}
        if ctx["tile_format"] == "jpeg":
            tile_key["quality"] = ctx["jpeg_quality"]
        tiles_valid = layout_reused and self.old.get("tile_key") == tile_key

        regions = tile_regions(layout["width"], layout["height"], ctx["tile_size"])
        names = [cache_tile_name(self.img_path, self.source_id, x, y, ctx["tile_format"]) for x, y, _, _ in regions]
        for name in self.old.get("tiles") or {}:
            if not tiles_valid or name not in names:
                self.counts["tiles_deleted"] += _unlink(tiles_dir / name)

        tiles = {}
        prepared = None
        for name, region in zip(names, regions, strict=True):
            path = tiles_dir / name
            if not (tiles_valid and name in (self.old.get("tiles") or {}) and path.is_file()):
                if prepared is None:
                    prepared = to_rgb_uchar(self.prepared(layout))
                write_tile_file(prepared, region, path, ctx["tile_format"], ctx["jpeg_quality"])
                self.counts["tiles_written"] += 1
            tiles[name] = list(region)

        source_image = self._update_source_image(file_info)
        status = "new"
        if self.old:
            status = "modified" if self.old.get("file", {}).get("sha256") != file_info["sha256"] else "reused"
        self.counts[status] += 1
        return {
            "source_id": self.source_id,
            "entry": {
                "file_name": self.img_info["file_name"],
                "file": file_info,
                "split": self.split,
                "layout": layout,
                "tile_key": tile_key,
                "tiles": tiles,
                "source_image": source_image,
            },
        }

    def _update_source_image(self, file_info: dict) -> dict | None:
        ctx = self.ctx
        path = ctx["cache_dir"] / SOURCE_IMAGES_DIR / f"{self.source_id}.png"
        old_source = self.old.get("source_image")
        if self.split != "test":
            if old_source:
                _unlink(path)
            return None
        key = {"sha256": file_info["sha256"], "resize": ctx["cfg"].resize, "version": CACHE_VERSION}
        if old_source and old_source.get("key") == key and path.is_file():
            return old_source
        temporary = _temporary(path)
        try:
            width, height = write_source_image(self.image(), ctx["cfg"].resize, temporary)
            os.replace(temporary, path)
        except BaseException:
            _unlink(temporary)
            raise
        self.counts["source_images_written"] += 1
        return {"key": key, "width": width, "height": height}


def _sweep_orphans(cache_dir: Path, manifest_images: dict) -> int:
    """Delete cached files the manifest does not reference (interrupted runs, failed or removed images)."""
    tiles = {name for entry in manifest_images.values() for name in entry["tiles"]}
    sources = {f"{source_id}.png" for source_id, entry in manifest_images.items() if entry.get("source_image")}
    removed = 0
    for directory, referenced in ((cache_dir / TILES_DIR, tiles), (cache_dir / SOURCE_IMAGES_DIR, sources)):
        for path in directory.iterdir():
            if path.is_file() and path.name not in referenced:
                path.unlink()
                removed += 1
    return removed


def named_annotations(raw_anns: list[dict], names: dict[int, str]) -> list[dict]:
    """Raw annotations with ``category_id`` replaced by the lowercase category name (resolved later by the view)."""
    return [{**ann, "category_id": names.get(ann["category_id"])} for ann in raw_anns]


def _tile_records(entry: dict, raw_anns: list[dict], names: dict[int, str], min_ratio: float, resize: float):
    """Tile annotations recomputed from the raw annotations (category by lowercase name), no pixels involved."""
    layout = entry["layout"]
    roi = ROI(*layout["roi"]) if layout["roi"] else None
    regions = [tuple(region) for region in entry["tiles"].values()]
    region_anns = tile_annotations(named_annotations(raw_anns, names), roi, layout["scale"], regions, min_ratio)
    origin_x = roi.x * layout["scale"] if roi else 0
    origin_y = roi.y * layout["scale"] if roi else 0
    records = []
    for name, (x, y, w, h), tile_anns in zip(entry["tiles"], regions, region_anns, strict=True):
        tile_anns = clean_annotation_boxes(copy.deepcopy(tile_anns), (w, h))
        records.append({
            "file_name": f"{TILES_DIR}/{name}",
            "width": w,
            "height": h,
            "tile_origin": [x + origin_x, y + origin_y],
            "tile_to_source_scale": resize / layout["scale"],
            "annotations": [
                {"iscrowd": 0, **{key: ann[key] for key in ANNOTATION_KEYS if key in ann},
                 "category": ann["category_id"]}
                for ann in tile_anns
            ],
        })
    return records


def update_source(input_dir: Path, cache_dir: Path, cfg, global_cfg, cached_cfg, *, full: bool = False) -> SourceCache:
    """Bring the tile cache of one source up to date and describe its tiles (with annotations) per original."""
    input_dir, cache_dir = Path(input_dir), Path(cache_dir)
    validate_ratios(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)
    coco = read_annotations(input_dir / "_annotations.coco.json")
    validate_coco(coco)
    file_names = [image["file_name"] for image in coco["images"]]
    if len(set(file_names)) != len(file_names):
        raise ValueError(f"Duplicate image file_name in {input_dir}.")
    source_ids = {image["id"]: image_source_id(input_dir / image["file_name"]) for image in coco["images"]}
    if len(set(source_ids.values())) != len(source_ids):
        raise ValueError(f"Duplicate images (same resolved path) in {input_dir}.")

    manifest = None if full else _load_manifest(cache_dir, input_dir)
    if manifest is None and cache_dir.exists():
        shutil.rmtree(cache_dir)
    previous = manifest["images"] if manifest else {}
    (cache_dir / TILES_DIR).mkdir(parents=True, exist_ok=True)
    (cache_dir / SOURCE_IMAGES_DIR).mkdir(exist_ok=True)

    split_key = {"seed": global_cfg.seed, "ratios": [cfg.train_ratio, cfg.val_ratio, cfg.test_ratio]}
    frozen = {}
    if manifest and manifest.get("split_key") == split_key:
        frozen = {
            image_id: previous[source_id]["split"]
            for image_id, source_id in source_ids.items() if source_id in previous
        }
    assignments = assign_splits(coco, tuple(split_key["ratios"]), global_cfg.seed, frozen)

    result = SourceCache(input_dir=input_dir, cache_dir=cache_dir, coco=coco)
    current = set(source_ids.values())
    result.stats["deleted"] = sum(source_id not in current for source_id in previous)

    names = {cat["id"]: str(cat["name"]).lower() for cat in coco["categories"]}
    grouped = group_annotations_by_image(coco)
    ctx = {
        "input_dir": input_dir, "cache_dir": cache_dir, "cfg": cfg,
        "tile_size": TILE_SIZE_MAPPER[global_cfg.model_type],
        "tile_format": cached_cfg.tile_format, "jpeg_quality": cached_cfg.jpeg_quality,
    }
    jobs = [
        _ImageJob(ctx, image, grouped.get(image["id"], []), assignments[image["id"]],
                  previous.get(source_ids[image["id"]]))
        for image in coco["images"]
    ]
    print(f"Actualizando la caché de tiles de {len(jobs)} imagenes en {cache_dir}.")
    with ThreadPoolExecutor(max_workers=cfg.n_jobs) as executor:
        outputs = list(executor.map(lambda job: job.run(), jobs))

    entries = {}
    for job, output in zip(jobs, outputs, strict=True):
        result.stats.update(job.counts)
        if output is None:
            result.stats["failed"] += 1
            continue
        entry = output["entry"]
        entries[output["source_id"]] = entry
        raw_anns = grouped.get(job.img_info["id"], [])
        source_image = None
        if entry["source_image"]:
            source_image = source_image_record(
                output["source_id"], job.img_info, named_annotations(raw_anns, names), cfg.resize,
                entry["source_image"]["width"], entry["source_image"]["height"],
            )
        result.images.append({
            "source_id": output["source_id"],
            "split": entry["split"],
            "tiles": _tile_records(entry, raw_anns, names, cfg.min_retained_dimension_ratio, cfg.resize),
            "source_image": source_image,
        })

    _write_json_atomic(cache_dir / MANIFEST_NAME, {
        "version": CACHE_VERSION,
        "source": str(input_dir.resolve()),
        "split_key": split_key,
        "images": entries,
    })
    result.stats["orphans_removed"] += _sweep_orphans(cache_dir, entries)
    return result


def update_cache(sources: list[Path], cache_dir: Path, cfg, global_cfg, cached_cfg, *, full: bool = False):
    """Update the cache of every source, in ``data.sources`` order (the fusion order of the view)."""
    cache_dir = Path(cache_dir)
    results = []
    for source in sources:
        result = update_source(source, source_cache_dir(cache_dir, source), cfg, global_cfg, cached_cfg, full=full)
        print(f"Caché {result.cache_dir.name}: {dict(result.stats)}")
        results.append(result)
    used = {result.cache_dir.name for result in results}
    sources_dir = cache_dir / "sources"
    stale = sorted(
        path.name for path in sources_dir.iterdir() if path.is_dir() and path.name not in used
    ) if sources_dir.is_dir() else []
    if stale:
        print(f"Cachés de fuentes que ya no están en data.sources (se conservan; bórralas si no hacen falta): {stale}")
    return results
