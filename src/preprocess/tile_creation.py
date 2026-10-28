import gc
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .image import open_image
from .roi import detect_roi
from .geometry import build_geometry_index, calculate_overlap
from .annotations import (
    group_annotations_by_image,
    read_annotations,
    transform_annotations,
    write_annotations,
)
from ..params import get_global_config, get_preprocess_config


_RAM_SEMAPHORE = threading.Semaphore(4)
TILE_SIZE_MAPPER = {"nano": 384, "small": 512, "medium": 576, "large": 704}


def _process_image(
    img_info: dict,
    raw_anns: list[dict],
    input_dir: Path,
    output_dir: Path,
    tile_size: int,
    cfg,
    save_empty: bool,
    save_source_image: bool,
) -> tuple[list[dict], list[dict], dict | None]:
    with _RAM_SEMAPHORE:
        try:
            return _process_image_inner(
                img_info, raw_anns, input_dir, output_dir, tile_size,
                cfg, save_empty, save_source_image,
            )
        finally:
            gc.collect()


def _process_image_inner(
    img_info, raw_anns, input_dir, output_dir, tile_size, cfg,
    save_empty, save_source_image,
):
    img_path = input_dir / img_info["file_name"]
    source_id = uuid5(NAMESPACE_URL, str(img_path.resolve())).hex
    img = open_image(str(img_path))
    orig_w, orig_h = img.width, img.height
    source_image = None
    if save_source_image:
        full = img.resize(cfg.resize) if cfg.resize != 1.0 else img
        filename = str(Path("source_images") / f"{source_id}.png")
        full.write_to_file(str(output_dir / filename))
        source_image = {
            "id": source_id,
            "file_name": filename,
            "original_file_name": img_info["file_name"],
            "width": full.width,
            "height": full.height,
            "has_defects": bool(raw_anns),
            "annotations": transform_annotations(raw_anns, None, cfg.resize),
        }
        del full

    roi = detect_roi(str(img_path), orig_w, orig_h) if cfg.apply_roi else None
    if roi is not None:
        img = img.crop(roi.x, roi.y, roi.w, roi.h)
    if cfg.resize != 1.0:
        img = img.resize(cfg.resize)
    transformed_anns = transform_annotations(raw_anns, roi, cfg.resize)
    geometries, ann_map, tree = build_geometry_index(transformed_anns)
    images, annotations = [], []
    image_id = ann_id = 1
    origin_x = roi.x * cfg.resize if roi else 0
    origin_y = roi.y * cfg.resize if roi else 0
    xs = sorted({min(x, max(0, img.width - tile_size))
                 for x in range(0, img.width, tile_size)})
    ys = sorted({min(y, max(0, img.height - tile_size))
                 for y in range(0, img.height, tile_size)})
    width, height = min(tile_size, img.width), min(tile_size, img.height)
    for y in ys:
        for x in xs:
            tile_anns, next_ann_id = ([], ann_id) if tree is None else calculate_overlap(
                x, y, width, height, image_id, ann_id,
                geometries, ann_map, tree, cfg.min_retained_dimension_ratio,
            )
            if not tile_anns and not save_empty:
                continue
            filename = f"{img_path.stem}_{source_id}_tile_{x}_{y}.png"
            tile = img.crop(x, y, width, height)
            tile.write_to_file(str(output_dir / filename))
            del tile
            images.append({
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
                "source_image_id": source_id,
                "tile_origin": [x + origin_x, y + origin_y],
            })
            annotations.extend(tile_anns)
            image_id += 1
            ann_id = next_ann_id
    return images, annotations, source_image


def create_tiles(
    output_dir: Path,
    input_dir: Path | None = None,
    *,
    annotations: dict | None = None,
    saving_prob: float | None = None,
    save_source_images: bool = False,
) -> None:
    global_cfg = get_global_config()
    cfg = get_preprocess_config()
    probability = cfg.saving_prob if saving_prob is None else saving_prob
    if not 0 <= probability <= 1:
        raise ValueError("saving_prob debe estar en [0, 1].")
    input_dir = Path(input_dir) if input_dir is not None else global_cfg.data_src1
    output_dir = Path(output_dir)
    coco = annotations if annotations is not None else read_annotations(
        input_dir / "_annotations.coco.json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_source_images:
        (output_dir / "source_images").mkdir(exist_ok=True)
    grouped = group_annotations_by_image(coco)
    rng = random.Random(global_cfg.seed)
    # Decide before scheduling workers so sampling is independent of thread order.
    jobs = [(image, rng.random() < probability) for image in coco["images"]]
    final_images, final_annotations, sources = [], [], []
    print(f"Creando tiles de {len(jobs)} imagenes en {output_dir}.")
    with ThreadPoolExecutor(max_workers=cfg.n_jobs) as executor:
        futures = [
            executor.submit(
                _process_image, image, grouped.get(image["id"], []),
                input_dir, output_dir, TILE_SIZE_MAPPER[global_cfg.model_type],
                cfg, save_empty, save_source_images,
            )
            for image, save_empty in jobs
        ]
        for future in futures:
            images, anns, source = future.result()
            local_ids = {}
            for image in images:
                local_ids[image["id"]] = len(final_images) + 1
                image["id"] = len(final_images) + 1
                final_images.append(image)
            for ann in anns:
                ann["id"] = len(final_annotations) + 1
                ann["image_id"] = local_ids[ann["image_id"]]
                final_annotations.append(ann)
            if source is not None:
                sources.append(source)
    result = {
        **coco, "images": final_images, "annotations": final_annotations,
        "source_images": sources,
    }
    write_annotations(output_dir / "_annotations.coco.json", result)
