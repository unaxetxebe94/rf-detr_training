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
            in_dir_path=data_src if resize == 1.0 else str(Path("data", f"{task_name}_formatted", "resized_temporal")),  # Si no se aplica resize, el input_dir es el directyorio de imágenes original
            out_dir_path=str(Path("data", f"{task_name}_formatted")),
            tile_size=tile_size_mapper[model_type],
            saving_prob=saving_prob,
            n_jobs=os.cpu_count() // 2
        )
        splitter = Splitter(
            dataset_dir=str(Path("data", f"{task_name}_formatted")),
            output_dir=str(Path("data", f"{task_name}_formatted")),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            image_action="move"
        )
        augmenter = Augmenter(
            input_dir=str(Path("data", f"{task_name}_formatted", "train")),
            output_dir=str(Path("data", f"{task_name}_formatted", "train")),
            augmentations_per_image=augmentations_per_image,
            max_transforms_per_sample=max_transforms_per_sample,
            seed=seed
        )
        label_corrector = LabelCorrector(
            dataset_path=str(Path("data", f"{task_name}_formatted"))
        )

        # Ejecutamos el pipeline
        if (resize != 1.0): resizer.run()
        tile_creator.run()
        if (resize != 1.0): shutil.rmtree(Path("data", f"{task_name}_formatted", "resized_temporal"))  # Se elimina la carpeta temporal de las imágenes redimensionadas
        splitter.run()
        augmenter.run()
        label_corrector.run()

    # Guardamos un mapping de cat_id --> cat_name para el test
    def save_mapping():
        dataset_train_path = Path("data", f"{task_name}_formatted", "train") if requires_preprocess else Path(data_src, "train")
        
        # Obtenemos los gts del entrenamiento para obtener las categorÃ­as
        annotations_path = Path(dataset_train_path, "_annotations.coco.json")
        with open(annotations_path, mode="r") as f:
            coco = json.load(f)
        categories = coco["categories"]
        output = {}
        
        # Simplemente conseguimos el mapeo de cat_id --> cat_name
        for cat in categories:
            if cat["id"] not in output:
                output[cat["id"]] = cat["name"]

        # Escribimos el mapeo en disco para luego poder leerlo desde el test
        output_path = Path("data", "temp", "category_map.json")
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, mode="w") as f:
            json.dump(output, f)

        logger.debug("Se ha terminado el preprocesado")

    save_mapping()