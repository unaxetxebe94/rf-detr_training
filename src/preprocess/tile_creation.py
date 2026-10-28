import os
import gc
import random
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .image import open_image
from .roi import ROI, detect_roi
from .geometry import build_geometry_index, calculate_overlap
from .annotations import read_annotations, transform_annotations, write_annotations
from ..params import GlobalConfig, get_global_config, PreprocessConfig, get_preprocess_config


_MAX_IMAGES_IN_RAM = 4
_RAM_SEMAPHORE = threading.Semaphore(max(1, _MAX_IMAGES_IN_RAM))
_BLACK_THRESHOLD = 10
TILE_SIZE_MAPPER = {
    "nano": 384,
    "small": 512,
    "medium": 576,
    "large": 704
}


def _process_image_inner(
    img_info: dict,
    anns: dict,
    start_img_id: int,
    start_ann_id: int,
    img_path: str,
    img_name: str,
    global_cfg: GlobalConfig,
    preprocess_cfg: PreprocessConfig,
    output_dir: Path
) -> tuple[list, list, int, int]:
    img = None
    try:
        # ── 1. Load ──────────────────────────────────────────────────
        try:
            img = open_image(img_path)
        except Exception as e:
            print(f"Cannot open {img_path}: {e}")
            return [], [], start_img_id, start_ann_id

        orig_w, orig_h = img.width, img.height
        base_name = os.path.splitext(img_name)[0]

        # ── 2. Optional ROI crop (in memory) ─────────────────────────
        roi: ROI | None = None
        if preprocess_cfg.apply_roi:
            roi = detect_roi(img_path, orig_w, orig_h)
            cropped = img.crop(roi.x, roi.y, roi.w, roi.h)
            del img           # release original before allocating resize
            img = cropped

        # ── 3. Optional resize (in memory) ───────────────────────────
        f = preprocess_cfg.resize
        if f != 1.0:
            resized = img.resize(f)
            del img           # release pre-resize copy
            img = resized

        img_w, img_h = img.width, img.height

        # ── 4. Transform annotations ──────────────────────────────────
        img_id = img_info.get("id")
        raw_anns = [
            a for a in anns.get("annotations", []) if a.get("image_id") == img_id
        ]
        transformed_anns = transform_annotations(raw_anns, roi, f)

        if not transformed_anns:
            print(f"Sin anotaciones para {img_name}, se omite.")
            return [], [], start_img_id, start_ann_id

        # ── 5. Build spatial index ────────────────────────────────────
        geometries, ann_map, tree = build_geometry_index(transformed_anns)
        if not tree:
            return [], [], start_img_id, start_ann_id

        new_images, new_annotations = [], []
        curr_img_id = start_img_id
        curr_ann_id = start_ann_id

        # ── 6. Tile ───────────────────────────────────────────────────
        tile_size = TILE_SIZE_MAPPER[global_cfg.model_type]
        for tile_y in range(0, img_h, tile_size):
            for tile_x in range(0, img_w, tile_size):
                actual_x = min(tile_x, max(0, img_w - tile_size))
                actual_y = min(tile_y, max(0, img_h - tile_size))
                tile_w = min(tile_size, img_w)
                tile_h = min(tile_size, img_h)

                tile_anns, next_ann_id = calculate_overlap(
                    actual_x, actual_y, tile_w, tile_h,
                    curr_img_id, curr_ann_id,
                    geometries, ann_map, tree, preprocess_cfg.min_retained_dimension_ratio
                )

                if not tile_anns and preprocess_cfg.saving_prob < random.random():
                    continue

                final_tile_name = (
                    f"{base_name}_tile_{actual_x}_{actual_y}.png"
                )
                final_tile_path = os.path.join(
                    output_dir, final_tile_name
                )

                try:
                    tile_img = img.crop(actual_x, actual_y, tile_w, tile_h)

                    if tile_img.avg() < _BLACK_THRESHOLD:
                        del tile_img
                        continue

                    tile_img.write_to_file(final_tile_path)
                    del tile_img   # free each tile after writing
                except Exception as e:
                    print(
                        f"Error al extraer tile ({tile_x},{tile_y}) "
                        f"de {img_name}: {e}"
                    )
                    continue

                new_images.append(
                    {
                        "id": curr_img_id,
                        "file_name": final_tile_name,
                        "width": tile_w,
                        "height": tile_h,
                    }
                )
                new_annotations.extend(tile_anns)
                curr_ann_id = next_ann_id
                curr_img_id += 1

        return new_images, new_annotations, curr_img_id, curr_ann_id

    finally:
        # Always release the pyvips image and nudge the GC,
        # even if an exception occurs mid-processing.
        if img is not None:
            del img
        gc.collect()


def _process_image(
    img_info: dict,
    anns: dict,
    start_img_id: int,
    start_ann_id: int,
    global_cfg: GlobalConfig,
    preprocess_cfg: PreprocessConfig,
    output_dir: Path,
    input_dir: Path,
) -> tuple[list, list, int, int]:
    img_name = img_info["file_name"]
    img_path = os.path.join(input_dir, img_name)

    if not os.path.exists(img_path):
        print(f"Imagen no encontrada: {img_path}")
        return [], [], start_img_id, start_ann_id

    # ── Acquire RAM slot before any large allocation ─────────────────
    # This blocks the thread until a slot is free, preventing multiple
    # workers from simultaneously holding large decoded images in RAM.
    with _RAM_SEMAPHORE:
        return _process_image_inner(
            img_info, anns, start_img_id, start_ann_id, img_path, img_name, global_cfg, preprocess_cfg, output_dir
        )


def create_tiles(output_dir: Path, input_dir: Path | None = None) -> None:
    global_cfg: GlobalConfig = get_global_config()
    preprocess_cfg: PreprocessConfig = get_preprocess_config()
    input_dir = Path(input_dir) if input_dir is not None else global_cfg.data_src1
    output_dir = Path(output_dir)

    anns = read_annotations(input_dir / "_annotations.coco.json")

    images_info = anns.get("images", [])
    if not images_info:
        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        for idx, fname in enumerate(os.listdir(input_dir)):
            if Path(fname).suffix.lower() in valid_exts:
                images_info.append({"id": idx, "file_name": fname})

    output_dir.mkdir(parents=True, exist_ok=True)

    final_images: list[dict] = []
    final_annotations: list[dict] = []
    global_img_id = 1
    global_ann_id = 1

    print(
        f"Iniciando procesado de {len(images_info)} imágenes "
        f"con {preprocess_cfg.n_jobs} hilos "
        f"(resize={preprocess_cfg.resize}, crop={preprocess_cfg.apply_roi})..."
    )

    with ThreadPoolExecutor(max_workers=preprocess_cfg.n_jobs) as executor:
        futures = [
            executor.submit(
                _process_image, img_info, anns, 1, 1,
                global_cfg, preprocess_cfg, output_dir, input_dir,
            )
            for img_info in images_info
        ]
        for future in as_completed(futures):
            try:
                new_imgs, new_anns, _, _ = future.result()

                local_img_map: dict[int, int] = {}
                for img in new_imgs:
                    local_img_map[img["id"]] = global_img_id
                    img["id"] = global_img_id
                    final_images.append(img)
                    global_img_id += 1

                for ann in new_anns:
                    ann["id"] = global_ann_id
                    ann["image_id"] = local_img_map[ann["image_id"]]
                    final_annotations.append(ann)
                    global_ann_id += 1

            except Exception as e:
                print(f"Error procesando imagen en worker: {e}")

    print("Procesado completado.")

    final_dict = {
        "images": final_images,
        "annotations": final_annotations,
        "categories": anns.get("categories", []),
    }

    write_annotations(output_dir / "_annotations.coco.json", final_dict)