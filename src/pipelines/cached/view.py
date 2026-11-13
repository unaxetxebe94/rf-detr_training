"""Fused Roboflow-style view (``train/valid/test`` + ``_annotations.coco.json``) over the tile cache.

Images are NTFS/POSIX hardlinks to the cached tiles: building the view is instant and takes no extra
space (falls back to copies when the cache lives on another volume). Every run only rewrites the
JSONs, so changing classes, ratios or negative sampling never touches the pixels. With the ``raw``
tile format, ``valid``/``test`` are materialised as PNG because the evaluation (``src.test``) opens
them with PIL; ``train`` keeps the ``.npy`` tiles, which only the cached dataset reads.
"""

import hashlib
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ...preprocess.annotations import DATASET_INFO, DATASET_LICENSES, SPLITS, read_annotations, write_annotations
from ...preprocess.incremental import _unlink, _write_json_atomic
from ..streaming.plan import build_categories
from .cache import SOURCE_IMAGES_DIR, SourceCache, read_tile_array

VIEW_VERSION = 1
VIEW_MANIFEST = "_cached_view.json"
ANNOTATIONS_FILE = "_annotations.coco.json"
VIEW_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".npy")


@dataclass
class DatasetView:
    dataset_dir: Path
    categories: list[dict]
    fingerprint: str
    summary: dict
    tile_format: str


def _view_file_name(cache_file: str, split: str, tile_format: str) -> str:
    name = Path(cache_file).name
    if tile_format == "raw" and split != "train":
        return str(Path(name).with_suffix(".png"))
    return name


def _map_annotations(anns: list[dict], category_ids: dict[str, int]) -> list[dict]:
    return [
        {**{key: value for key, value in ann.items() if key not in ("category", "id", "image_id")},
         "category_id": category_ids[ann["category"]]}
        for ann in anns if ann.get("category") in category_ids
    ]


def build_datasets(results: list[SourceCache], classes: list[str] | None, tile_format: str):
    """COCO JSON per split plus the cache file behind every view file. Pure: reads no pixels, writes nothing."""
    # supercategory "none" marks Roboflow's parent category, which src.test discards: real classes must not use it.
    categories = [
        {**cat, "supercategory": "defects"} for cat in build_categories([result.coco for result in results], classes)
    ]
    category_ids = {cat["name"]: cat["id"] for cat in categories}
    datasets = {
        split: {
            "info": dict(DATASET_INFO), "licenses": list(DATASET_LICENSES),
            "categories": [dict(cat) for cat in categories], "images": [], "annotations": [], "source_images": [],
        }
        for split in SPLITS
    }
    files: dict[str, dict[str, Path]] = {split: {} for split in SPLITS}
    for result in results:
        for original in result.images:
            split = original["split"]
            coco = datasets[split]
            for tile in original["tiles"]:
                file_name = _view_file_name(tile["file_name"], split, tile_format)
                if file_name in files[split]:
                    raise ValueError(f"Tile duplicado en la vista {split}: {file_name}.")
                files[split][file_name] = result.cache_dir / tile["file_name"]
                anns = _map_annotations(tile["annotations"], category_ids)
                image_id = len(coco["images"]) + 1
                coco["images"].append({
                    "id": image_id, "file_name": file_name, "width": tile["width"], "height": tile["height"],
                    "source_image_id": original["source_id"], "tile_origin": tile["tile_origin"],
                    "tile_to_source_scale": tile["tile_to_source_scale"], "has_defects": bool(anns),
                })
                for ann in anns:
                    coco["annotations"].append({**ann, "id": len(coco["annotations"]) + 1, "image_id": image_id})
            source = original["source_image"]
            if source is not None and split == "test":
                anns = [
                    {**{key: value for key, value in ann.items() if key not in ("id", "image_id", "category_id")},
                     "category_id": category_ids[ann["category_id"]]}
                    for ann in source["annotations"] if ann.get("category_id") in category_ids
                ]
                coco["source_images"].append({**source, "annotations": anns, "has_defects": bool(anns)})
                files[split][source["file_name"]] = result.cache_dir / SOURCE_IMAGES_DIR / f"{source['id']}.png"
    return categories, datasets, files


def dataset_fingerprint(datasets: dict[str, dict], tile_format: str) -> str:
    content = json.dumps(
        {"version": VIEW_VERSION, "tile_format": tile_format, "datasets": {split: datasets[split] for split in SPLITS}},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def dataset_summary(datasets: dict[str, dict]) -> dict:
    summary = {}
    for split in SPLITS:
        coco = datasets[split]
        positives = sum(image.get("has_defects", False) for image in coco["images"])
        summary.update({
            f"{split}/tiles": len(coco["images"]),
            f"{split}/positive_tiles": positives,
            f"{split}/negative_tiles": len(coco["images"]) - positives,
            f"{split}/annotations": len(coco["annotations"]),
            f"{split}/originals": len({image["source_image_id"] for image in coco["images"]}),
        })
    summary["test/source_images"] = len(datasets["test"]["source_images"])
    return summary


def _same_file(source: Path, destination: Path) -> bool:
    try:
        return os.path.samefile(source, destination)
    except OSError:
        return False


def _materialise(source: Path, destination: Path) -> str:
    """Make ``destination`` show ``source``'s content. Never writes through an existing file (it may be a hardlink)."""
    if destination.suffix.lower() == ".png" and source.suffix.lower() == ".npy":
        try:
            if destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
                return "kept"
        except FileNotFoundError:
            pass
        temporary = destination.with_name(f".{destination.stem}.tmp.png")
        try:
            Image.fromarray(read_tile_array(source)).save(temporary)
            os.replace(temporary, destination)
        except BaseException:
            _unlink(temporary)
            raise
        return "converted"
    if _same_file(source, destination):
        return "kept"
    try:
        stat, source_stat = destination.stat(), source.stat()
        if (stat.st_size, stat.st_mtime_ns) == (source_stat.st_size, source_stat.st_mtime_ns):
            return "kept"
        destination.unlink()
    except FileNotFoundError:
        pass
    try:
        os.link(source, destination)
        return "linked"
    except OSError:
        temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        except BaseException:
            _unlink(temporary)
            raise
        return "copied"


def _remove_unreferenced(split_dir: Path, referenced: set[str]) -> int:
    removed = 0
    for directory in (split_dir, split_dir / SOURCE_IMAGES_DIR):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.name == ANNOTATIONS_FILE:
                continue
            relative = path.relative_to(split_dir).as_posix()
            stale_image = path.suffix.lower() in VIEW_FILE_EXTENSIONS and relative not in referenced
            if stale_image or (path.name.startswith(".") and ".tmp" in path.name):
                path.unlink()
                removed += 1
    return removed


def build_view(results: list[SourceCache], dataset_dir: Path, classes: list[str] | None, tile_format: str,
               *, n_jobs: int = 8) -> tuple[DatasetView, dict]:
    """Write the fused view in ``dataset_dir`` and return it with per-run statistics."""
    dataset_dir = Path(dataset_dir)
    categories, datasets, files = build_datasets(results, classes, tile_format)
    stats = Counter()
    # Without its manifest the view counts as unfinished, so --skip-update refuses a half-built view.
    _unlink(dataset_dir / VIEW_MANIFEST)
    for split in SPLITS:
        split_dir = dataset_dir / split
        (split_dir / SOURCE_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
        stats["removed"] += _remove_unreferenced(split_dir, set(files[split]))
        jobs = [(source, split_dir / file_name) for file_name, source in files[split].items()]
        with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as executor:
            stats.update(executor.map(lambda job: _materialise(*job), jobs))
        write_annotations(split_dir / ANNOTATIONS_FILE, datasets[split])
    view = DatasetView(
        dataset_dir=dataset_dir, categories=categories, fingerprint=dataset_fingerprint(datasets, tile_format),
        summary=dataset_summary(datasets), tile_format=tile_format,
    )
    _write_json_atomic(dataset_dir / VIEW_MANIFEST, {
        "version": VIEW_VERSION, "fingerprint": view.fingerprint, "tile_format": tile_format,
        "categories": categories, "summary": view.summary,
    })
    if stats["copied"]:
        print(f"Aviso: {stats['copied']} tiles copiados en vez de enlazados (¿caché en otro volumen?).")
    return view, dict(stats)


def load_view(dataset_dir: Path) -> DatasetView:
    """Previously built view (``--skip-update``), checking its JSONs still match the recorded fingerprint."""
    dataset_dir = Path(dataset_dir)
    try:
        manifest = read_annotations(dataset_dir / VIEW_MANIFEST)
    except (OSError, ValueError) as error:
        raise FileNotFoundError(
            f"No hay una vista del pipeline cached en {dataset_dir}; ejecuta sin --skip-update."
        ) from error
    if manifest.get("version") != VIEW_VERSION:
        raise ValueError(f"La vista de {dataset_dir} es de otra versión; ejecuta sin --skip-update.")
    datasets = {split: read_annotations(dataset_dir / split / ANNOTATIONS_FILE) for split in SPLITS}
    fingerprint = dataset_fingerprint(datasets, manifest["tile_format"])
    if fingerprint != manifest["fingerprint"]:
        raise ValueError(
            f"Los JSON de {dataset_dir} no coinciden con {VIEW_MANIFEST} (¿los ha modificado otro pipeline?); "
            "ejecuta sin --skip-update."
        )
    return DatasetView(
        dataset_dir=dataset_dir, categories=manifest["categories"], fingerprint=fingerprint,
        summary=dataset_summary(datasets), tile_format=manifest["tile_format"],
    )
