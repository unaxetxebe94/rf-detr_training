# train.py
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
import yaml
import logging
from pathlib import Path
from datetime import datetime
from utils import set_seed
from logger import get_logger

if __name__ == "__main__":
    logger = get_logger(__name__, level=logging.DEBUG)

    # 1. Carga segura
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    set_seed(params["seed"])

    tp = params["train"]
    model_type = params["model-type"]

    # Accedemos a las rutas desde 'tp'
    now = datetime.now().timestamp()
    dataset_dir = Path("data", f"{params['task-name']}_formatted")
    output_dir = Path("trainings", params["task-name"])
    run = f"{now}_{params['task-name']}"

    # SelecciÃ³n de modelo (simplificada)
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
        run = run,
        mlflow = tp["mlflow"],
        clearml = tp["clearml"],
        run_test = tp["run_test"],
        eval_max_dets = tp["eval_max_dets"] 
    )
    logger.info("Se ha entrenado el modelo")

    # Guardamos el nombre del run para el test
    import json
    run_name_path = Path(output_dir.parent, "temp", "run_info.json")
    with open(run_name_path, "w") as f:
        json.dump({"run_name": run}, f)
    logger.info("Se ha escrito run_info")
    