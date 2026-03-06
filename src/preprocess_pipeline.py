import os

os.environ["PATH"] = r"C:\Program Files\vips-dev-8.17\bin;" + os.environ["PATH"]

import yaml
import json
import shutil
import logging
from pathlib import Path

from utils import set_seed, read_params, save_mapping
from preprocess_stage.resizer import Resizer
from preprocess_stage.tile_creator import TileCreator
from preprocess_stage.splitter import Splitter
from preprocess_stage.augmenter import Augmenter
from preprocess_stage.label_corrector import LabelCorrector
from logger import get_logger

logger = get_logger(__name__, level=logging.DEBUG)


def is_dataset_formatted(dataset_dir: Path) -> bool:
    if os.path.exists(formatted_dataset_dir):
        train_exists = os.path.exists(formatted_dataset_dir / "train")
        test_exists = os.path.exists(formatted_dataset_dir / "test")
        valid_exists = os.path.exists(formatted_dataset_dir / "valid")
        return train_exists and test_exists and valid_exists
    else: 
        logger.warning("No se encontró la carpeta que debería contener el dataset formateado.")

if __name__ == "__main__":

    logger.info("\n\n\n====================== INICIANDO PIPELINE ======================")

    params = read_params()
    seed=params["seed"]
    set_seed(seed=seed)

    # Obtenemos los parámetros de preprocesamiento de params.yaml
    input_folder = params["data-src1"]
    requires_preprocess = params["preprocess"]["requires-preprocess"]
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
    model_type = params["model-type"].lower()
    formatted_dataset_dir = Path("data", "formatted")


    # Inicializamos los pasos del pipeline si se requiere
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
            input_folder=input_folder,
            output_folder=str(Path("data", "formatted", "resized_temporal")),
            resize_factor=resize,
            apply_roi=apply_roi
        )
        tile_creator = TileCreator(
            in_dir_path=input_folder if resize == 1.0 else str(Path("data", "formatted", "resized_temporal")),
            out_dir_path=str(formatted_dataset_dir),
            tile_size=tile_size_mapper[model_type],
            saving_prob=saving_prob,
            n_jobs=os.cpu_count() // 2
        )
        splitter = Splitter(
            dataset_dir=str(formatted_dataset_dir),
            output_dir=str(formatted_dataset_dir),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            image_action="move"
        )
        augmenter = Augmenter(
            input_dir=str(formatted_dataset_dir / "train"),
            output_dir=str(formatted_dataset_dir / "train"),
            augmentations_per_image=augmentations_per_image,
            max_transforms_per_sample=max_transforms_per_sample,
            seed=seed
        )
        label_corrector = LabelCorrector(
            dataset_path=str(formatted_dataset_dir)
        )
        # Ejecutamos el pipeline
        if (resize != 1.0): resizer.run()
        tile_creator.run()
        if (resize != 1.0): shutil.rmtree(Path("data", "formatted", "resized_temporal"))
        splitter.run()
        augmenter.run()
        label_corrector.run()
    else:
        if (is_dataset_formatted(formatted_dataset_dir)): 
            logger.warning("No se ha encontrado el dataset formateado donde debería estar. Se procederá a fusionar los datasets de entrada asumiendo que ya están formateados y preparados para ello.")    
            save_mapping()
            logger.info("Se ha guardado el category_map para el test")
        else:
            # Si no se requiere preprocesar, se asumirá que los datasets están formateados y preparados para unirlos
            if not (is_dataset_formatted(params["data-src1"]) and is_dataset_formatted(params["data-src2"])):
                raise Exception(f"No se han encontrado los datasets formateados en los paths especificados --> SRC1: {params['data-src1']} - SRC2: {params['data-src2']}")
            
    logger.info("Se ha terminado el preprocesado de los datasets. Se procede a fusionarlos.")