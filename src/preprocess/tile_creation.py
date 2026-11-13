import gc
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from ..pipelines.disk.params import get_global_config, get_preprocess_config
from .annotations import (
    group_annotations_by_image,
    read_annotations,
    transform_annotations,
    write_annotations,
)
from .geometry import build_geometry_index, calculate_overlap
from .image import open_image
from .roi import ROI, detect_roi

_RAM_SEMAPHORE = threading.Semaphore(4)
TILE_SIZE_MAPPER = {"nano": 384, "small": 512, "medium": 576, "large": 704}


def image_source_id(img_path: Path) -> str:
    return uuid5(NAMESPACE_URL, str(Path(img_path).resolve())).hex


def sample_value(seed: int, source_id: str) -> float:
    """Uniform value in [0, 1) fixed per original image, independent of processing order."""
    digest = hashlib.sha256(f"{seed}:{source_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def tile_file_name(img_path: Path, source_id: str, x: int, y: int) -> str:
    return f"{Path(img_path).stem}_{source_id}_tile_{x}_{y}.png"


def _resize_scale(img, tile_size: int, resize: float) -> float:
    """Factor de escala de pyvips (`Image.resize(scale)`).

    Si la imagen es menor que el tamaño de tile en alguna dimensión, se amplifica
    hasta que ambas dimensiones alcancen el tile; si no, se aplica el `resize`
    global. El mismo factor se usa para escalar las anotaciones.
    """
    if img.height < tile_size or img.width < tile_size:
        return max(tile_size / img.width, tile_size / img.height)
    return resize


def prepare_image_for_tiles(img, img_path, tile_size, cfg):
    roi = detect_roi(str(img_path), img.width, img.height) if cfg.apply_roi else None
    if roi is not None:
        img = img.crop(roi.x, roi.y, roi.w, roi.h)
    scale = _resize_scale(img, tile_size, cfg.resize)
    if scale != 1.0:
        img = img.resize(scale)
    return img, roi, scale


def apply_layout(img, roi: ROI | None, scale: float):
    """Repeat the crop/resize of `prepare_image_for_tiles` without detecting the ROI again."""
    if roi is not None:
        img = img.crop(roi.x, roi.y, roi.w, roi.h)
    if scale != 1.0:
        img = img.resize(scale)
    return img


def tile_regions(width: int, height: int, tile_size: int):
    xs = sorted({min(x, max(0, width - tile_size)) for x in range(0, width, tile_size)})
    ys = sorted({min(y, max(0, height - tile_size)) for y in range(0, height, tile_size)})
    tile_w, tile_h = min(tile_size, width), min(tile_size, height)
    return [(x, y, tile_w, tile_h) for y in ys for x in xs]


def iter_tile_regions(img, tile_size):
    yield from tile_regions(img.width, img.height, tile_size)


def tile_annotations(raw_anns, roi: ROI | None, scale: float, regions, min_ratio: float):
    """Annotations clipped to each region (tile coordinates), without reading pixels."""
    transformed = transform_annotations(raw_anns, roi, scale)
    geometries, ann_map, tree = build_geometry_index(transformed)
    if tree is None:
        return [[] for _ in regions]
    return [
        calculate_overlap(x, y, w, h, 0, 0, geometries, ann_map, tree, min_ratio)[0]
        for x, y, w, h in regions
    ]


def write_tile(prepared_img, region, path: Path) -> None:
    x, y, w, h = region
    prepared_img.crop(x, y, w, h).write_to_file(str(path))


def write_source_image(img, resize: float, path: Path) -> tuple[int, int]:
    """Full original image, resized without ROI, used for whole-door evaluation."""
    full = img.resize(resize) if resize != 1.0 else img
    path.parent.mkdir(parents=True, exist_ok=True)
    full.write_to_file(str(path))
    return full.width, full.height


def source_image_record(source_id, img_info, raw_anns, resize, width, height) -> dict:
    return {
        "id": source_id,
        "file_name": str(Path("source_images") / f"{source_id}.png"),
        "original_file_name": img_info["file_name"],
        "width": width,
        "height": height,
        "has_defects": bool(raw_anns),
        "annotations": transform_annotations(raw_anns, None, resize),
    }


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
    source_id = image_source_id(img_path)
    try:
        img = open_image(str(img_path))
    except Exception:
        print(f"Failed to open {img_path}")
        return [], [], None
    source_image = None
    if save_source_image:
        source_image = source_image_record(source_id, img_info, raw_anns, cfg.resize, 0, 0)
        source_image["width"], source_image["height"] = write_source_image(
            img, cfg.resize, output_dir / source_image["file_name"],
        )

    img, roi, scale = prepare_image_for_tiles(img, img_path, tile_size, cfg)
    regions = tile_regions(img.width, img.height, tile_size)
    region_anns = tile_annotations(raw_anns, roi, scale, regions, cfg.min_retained_dimension_ratio)
    images, annotations = [], []
    origin_x = roi.x * scale if roi else 0
    origin_y = roi.y * scale if roi else 0
    for region, tile_anns in zip(regions, region_anns, strict=True):
        if not tile_anns and not save_empty:
            continue
        x, y, width, height = region
        filename = tile_file_name(img_path, source_id, x, y)
        write_tile(img, region, output_dir / filename)
        image_id = len(images) + 1
        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
            "source_image_id": source_id,
            "tile_origin": [x + origin_x, y + origin_y],
            "tile_to_source_scale": cfg.resize / scale,
        })
        for ann in tile_anns:
            annotations.append({**ann, "id": len(annotations) + 1, "image_id": image_id})
    return images, annotations, source_image


def create_tiles(
    output_dir: Path,
    input_dir: Path | None = None,
    *,
    annotations: dict | None = None,
    saving_prob: float | None = None,
    save_source_images: bool = False,
) -> None:
    """Tile every image of a COCO dataset at once (non-incremental helper)."""
    global_cfg = get_global_config()
    cfg = get_preprocess_config()
    probability = cfg.saving_prob if saving_prob is None else saving_prob
    if not 0 <= probability <= 1:
        raise ValueError("saving_prob debe estar en [0, 1].")
    input_dir = Path(input_dir) if input_dir is not None else global_cfg.sources[0]
    output_dir = Path(output_dir)
    coco = annotations if annotations is not None else read_annotations(
        input_dir / "_annotations.coco.json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_source_images:
        (output_dir / "source_images").mkdir(exist_ok=True)
    grouped = group_annotations_by_image(coco)
    jobs = [
        (image, sample_value(global_cfg.seed, image_source_id(input_dir / image["file_name"])) < probability)
        for image in coco["images"]
    ]
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
