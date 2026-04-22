import os
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
import yaml
import logging
from pathlib import Path
from datetime import datetime
from utils import set_seed, read_params
from logger import get_logger

if __name__ == "__main__":
    logger = get_logger(__name__, level=logging.DEBUG)

    # 1. Carga segura
    params = read_params()
    set_seed(params["seed"])

    tp = params["train"]  # training parameters
    model_type = params["model-type"]
    pp = params["preprocess"]  # preprocess parameters

    # Accedemos a las rutas desde 'tp'
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_dir = Path("data", "formatted")
    output_dir = Path("trainings", "training")
    run_name = f"[{now}]-saving_prob={pp['saving-prob']}-resize={pp['resize']}-apply_roi={pp['apply-roi']}-train_ratio={pp['train-ratio']}-test_ratio={pp['test-ratio']}-val_ratio={pp['val-ratio']}-augmentations_per_image={pp['augmentations-per-image']}-max_transformations_per_sample={pp['max-transforms-per-sample']}"
    # SelecciÃ³n de modelo
    models = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge
    }
    model = models[str.lower(model_type)]()
    logger.debug(f"Se ha iniciado un modelo {model_type}")

    # 2. Entrenamiento con las llaves corregidas
    model.train(
        lr = tp["lr"],
        lr_encoder = tp["lr_encoder"],
        batch_size = tp["batch_size"],
        grad_accum_steps = tp["grad_accum_steps"],
        epochs = tp["epochs"],           
        dataset_dir = str(dataset_dir),
        output_dir = str(output_dir),
        weight_decay = tp["weight_decay"],
        tensorboard = tp["tensorboard"],
        wandb = tp["wandb"],
        project = tp["project"],
        run = run_name,
        mlflow = tp["mlflow"],
        clearml = tp["clearml"],
        run_test = tp["run_test"],
        eval_max_dets = tp["eval_max_dets"] ,
        seed=params["seed"]
    )
    logger.info("Se ha entrenado el modelo")

    # Guardamos el nombre del run para el test
    import json
    run_name_path = Path(output_dir.parent, "temp", "run_info.json")
    os.makedirs(run_name_path.parent, exist_ok=True)
    with open(run_name_path, "w") as f:
        json.dump({"run_name": run_name}, f)
    logger.info("Se ha escrito run_info")
    