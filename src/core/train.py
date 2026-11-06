import json
import logging
from datetime import datetime
from pathlib import Path

from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

from core.logger import get_logger
from core.utils import read_params, set_seed

if __name__ == "__main__":
    logger = get_logger(__name__, level=logging.DEBUG)

    # 1. Carga segura
    params = read_params()
    set_seed(params["seed"])

    tp = params["train"]  # training parameters
    model_type = params["model-type"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        pp = params["preprocess"]  # preprocess parameters
        run_name = (
            f"[{now}]-saving_prob={pp['saving-prob']}-resize={pp['resize']}"
            f"-apply_roi={pp['apply-roi']}-train_ratio={pp['train-ratio']}"
            f"-test_ratio={pp['test-ratio']}-val_ratio={pp['val-ratio']}"
            f"-augmentations_per_image={pp['augmentations-per-image']}"
            f"-max_transformations_per_sample={pp['max-transforms-per-sample']}"
        )
    except KeyError as ke:
        print(f"Error al leer el preprocesado: {ke}")
        print("Se asume que se está entrenando desde Collab")
        run_name = f"[{now}]-CollabTraining"

    # Accedemos a las rutas desde 'tp'
    dataset_dir = Path(params["data"]["training-dataset-path"])
    output_dir = Path("trainings", "training")
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
    with open(dataset_dir / "train" / "_annotations.coco.json", encoding="utf-8") as f:
        categories = json.load(f)["categories"]
    with open(output_dir / "training_categories.json", "w", encoding="utf-8") as f:
        json.dump({"categories": categories}, f, ensure_ascii=False, indent=2)
