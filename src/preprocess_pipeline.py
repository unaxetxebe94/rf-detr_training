import os

os.environ["PATH"] = r"C:\Program Files\vips-dev-8.17\bin;" + os.environ["PATH"]

import yaml
import json
import shutil
import logging
from pathlib import Path

from utils import set_seed
from preprocess_stage.resizer import Resizer
from preprocess_stage.tile_creator import TileCreator
from preprocess_stage.splitter import Splitter
from preprocess_stage.augmenter import Augmenter
from preprocess_stage.label_corrector import LabelCorrector
from logger import get_logger

logger = get_logger(__name__, level=logging.DEBUG)


def _count_split_stats(dataset_dir: Path, splits: tuple = ("train", "val", "test")) -> dict:
    """
    Cuenta imágenes y anotaciones en cada split del dataset formateado.
    Devuelve un dict {split: {n_images, n_annotations}}.
    """
    stats = {}
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue

        # Contar imágenes
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".tiff"}
        n_images = sum(
            1 for f in split_dir.iterdir()
            if f.suffix.lower() in image_exts
        )

        # Contar anotaciones desde el COCO JSON si existe
        annotations_file = split_dir / "_annotations.coco.json"
        n_annotations = 0
        if annotations_file.exists():
            with open(annotations_file) as f:
                coco = json.load(f)
            n_annotations = len(coco.get("annotations", []))

        stats[split] = {"n_images": n_images, "n_annotations": n_annotations}
        logger.debug(f"Split '{split}': {n_images} imágenes, {n_annotations} anotaciones")

    return stats


def save_preprocess_info(params: dict, dataset_dir: Path, output_dir: Path) -> None:
    """
    Guarda un JSON con los parámetros de preprocesado y las estadísticas del
    dataset resultante. Este fichero lo lee process_results.py para logearlo en W&B.
    """
    preprocess_params = params.get("preprocess", {})

    # Estadísticas de los splits generados
    split_stats = _count_split_stats(dataset_dir)

    info = {
        "params": {
            "resize":                    preprocess_params.get("resize"),
            "apply_roi":                 preprocess_params.get("apply-roi"),
            "saving_prob":               preprocess_params.get("saving-prob"),
            "train_ratio":               preprocess_params.get("train-ratio"),
            "val_ratio":                 preprocess_params.get("val-ratio"),
            "test_ratio":                preprocess_params.get("test-ratio"),
            "augmentations_per_image":   preprocess_params.get("augmentations-per-image"),
            "max_transforms_per_sample": preprocess_params.get("max-transforms-per-sample"),
            "model_type":                params.get("model-type"),
            "seed":                      params.get("seed"),
            "data_src":                  params.get("data-src"),
            "task_name":                 params.get("task-name"),
        },
        "split_stats": split_stats,
    }

    output_path = output_dir / "preprocess_info.json"
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"preprocess_info.json guardado → {output_path}")


if __name__ == "__main__":

    # Leemos el archivo de parámetros del pipeline
    with open("params.yaml", mode="r") as f:
        params = yaml.safe_load(f)
    
    # Obtenemos los parámetros de preprocesamiento de params.yaml
    requires_preprocess = params["preprocess"]["requires-preprocess"]
    
    data_src = params["data-src"]
    resize = params["preprocess"]["resize"]
    saving_prob = params["preprocess"]["saving-prob"]
    apply_roi = params["preprocess"]["apply-roi"]
    task_name = params["task-name"]
    train_ratio = float(params["preprocess"]["train-ratio"])
    test_ratio = float(params["preprocess"]["test-ratio"])
    val_ratio = float(params["preprocess"]["val-ratio"])
    augmentations_per_image = params["preprocess"]["augmentations-per-image"]
    max_transforms_per_sample = params["preprocess"]["max-transforms-per-sample"]
    if train_ratio + val_ratio + test_ratio != 1.0: raise Exception("Los split ratios no suman 1.0!")
    seed = params["seed"]
    model_type = params["model-type"].lower()
    set_seed(seed=seed)

    dataset_dir = Path("data", f"{task_name}_formatted")
    output_dir  = Path("trainings", "temp")

    if requires_preprocess:
        # Mapeo de los tiles de cada tipo de modelo
        tile_size_mapper = {
            "nano": 384,
            "small": 512,
            "medium": 576,
            "large": 704
        }

        # Preparamos los tres pasos del pipeline
        resizer = Resizer(
            input_folder=data_src,
            output_folder=str(Path("data", f"{task_name}_formatted", "resized_temporal")),
            resize_factor=resize,
            apply_roi=apply_roi
        )
        tile_creator = TileCreator(
            in_dir_path=data_src if resize == 1.0 else str(Path("data", f"{task_name}_formatted", "resized_temporal")),
            out_dir_path=str(dataset_dir),
            tile_size=tile_size_mapper[model_type],
            saving_prob=saving_prob,
            n_jobs=os.cpu_count() // 2
        )
        splitter = Splitter(
            dataset_dir=str(dataset_dir),
            output_dir=str(dataset_dir),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            image_action="move"
        )
        augmenter = Augmenter(
            input_dir=str(dataset_dir / "train"),
            output_dir=str(dataset_dir / "train"),
            augmentations_per_image=augmentations_per_image,
            max_transforms_per_sample=max_transforms_per_sample,
            seed=seed
        )
        label_corrector = LabelCorrector(
            dataset_path=str(dataset_dir)
        )

        # Ejecutamos el pipeline
        if (resize != 1.0): resizer.run()
        tile_creator.run()
        if (resize != 1.0): shutil.rmtree(Path("data", f"{task_name}_formatted", "resized_temporal"))
        splitter.run()
        augmenter.run()
        label_corrector.run()

    # Guardamos un mapping de cat_id --> cat_name para el test
    def save_mapping():
        dataset_train_path = dataset_dir / "train" if requires_preprocess else Path(data_src, "train")
        
        annotations_path = dataset_train_path / "_annotations.coco.json"
        with open(annotations_path, mode="r") as f:
            coco = json.load(f)
        categories = coco["categories"]
        output = {}
        
        for cat in categories:
            if cat["id"] not in output:
                output[cat["id"]] = cat["name"]

        output_path = Path("data", "temp", "category_map.json")
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, mode="w") as f:
            json.dump(output, f)

        logger.debug("Se ha guardado el category_map")

    save_mapping()

    # ── Guardar info de preprocesado para W&B ────────────────────────────────
    # Se usa la ruta del dataset formateado si se ha preprocesado,
    # o el data-src original si el dataset ya venía dividido.
    stats_dir = dataset_dir if requires_preprocess else Path(data_src)
    save_preprocess_info(params, stats_dir, output_dir)

    logger.debug("Se ha terminado el preprocesado")