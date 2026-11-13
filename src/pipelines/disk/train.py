import datetime
import json
import logging
from dataclasses import asdict
from pathlib import Path

from rfdetr import RFDETR, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr.config import RFDETRBaseConfig, RFDETRLargeConfig, RFDETRMediumConfig, RFDETRNanoConfig, RFDETRSmallConfig
from rfdetr.config import TrainConfig as RFDETRTrainConfig
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

from .logger import get_logger
from .mlflow import get_mlflow_logger, log_dict_params, mlflow_run
from .params import TrainConfig, get_data_config, get_global_config, get_preprocess_config, get_train_config
from .seed import set_seed

MODELS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge
}

MODEL_CONFIGS = {
    "nano": RFDETRNanoConfig,
    "small": RFDETRSmallConfig,
    "medium": RFDETRMediumConfig,
    "large": RFDETRLargeConfig
}


def get_num_classes(dataset_dir: Path) -> int:
    anns_file = dataset_dir / "train" / "_annotations.coco.json"
    if not anns_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo 'classes.json' en {dataset_dir}")
    with open(anns_file) as f:
        anns = json.load(f)
    return len(anns["categories"])


def get_model_and_config(model_type: str, dataset_dir: Path) -> tuple[RFDETR, RFDETRBaseConfig]:
    model_type = model_type.lower()
    if model_type not in MODELS:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}. Opciones válidas: {list(MODELS.keys())}")
    return MODELS[model_type](), MODEL_CONFIGS[model_type](num_classes=get_num_classes(dataset_dir))


def get_rfdetr_train_cfg(dataset_dir: Path, output_dir: Path, training_params: TrainConfig) -> RFDETRTrainConfig:
    return RFDETRTrainConfig(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        lr=training_params.lr,
        lr_encoder=training_params.lr_encoder,
        batch_size=training_params.batch_size,
        grad_accum_steps=training_params.grad_accum_steps,
        epochs=training_params.epochs,
        weight_decay=training_params.weight_decay,
        tensorboard=training_params.tensorboard,
        # "cpu" (the RF-DETR default) silently picks Albumentations once rfdetr[augment] is installed.
        augmentation_backend="torchvision",
        mlflow=False
    )


def main(output_dir: Path | None = None) -> Path:
    with mlflow_run("train"):
        return _train(output_dir)


def _train(output_dir: Path | None) -> Path:
    logger = get_logger(__name__, level=logging.DEBUG)

    global_cfg = get_global_config()
    data_cfg = get_data_config()
    train_cfg = get_train_config()
    set_seed(global_cfg.seed)

    model_type = global_cfg.model_type
    dataset_dir = Path(data_cfg.training_dataset_path)
    output_dir = output_dir or Path("trainings", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    model_config = MODEL_CONFIGS[model_type](num_classes=get_num_classes(dataset_dir))

    rfdetr_train_cfg = get_rfdetr_train_cfg(dataset_dir, output_dir, train_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_dir / "train" / "_annotations.coco.json", encoding="utf-8") as f:
        categories = json.load(f)["categories"]
    with open(output_dir / "training_categories.json", "w", encoding="utf-8") as f:
        json.dump({"categories": categories}, f, indent=2)
    log_dict_params(asdict(global_cfg), prefix="global/")
    log_dict_params(asdict(get_preprocess_config()), prefix="preprocess/")
    log_dict_params(rfdetr_train_cfg.model_dump(mode="json"), prefix="train/")
    log_dict_params({"training/output_dir": str(output_dir.resolve())})
    trainer = build_trainer(rfdetr_train_cfg, model_config)
    trainer.loggers = [*trainer.loggers, get_mlflow_logger()]

    module = RFDETRModelModule(model_config, rfdetr_train_cfg)
    data_module = RFDETRDataModule(model_config, rfdetr_train_cfg)

    trainer.fit(module, data_module)
    checkpoint = output_dir / "checkpoint_best_total.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"El entrenamiento no ha generado el checkpoint: {checkpoint}")
    log_dict_params({"training/checkpoint": str(checkpoint.resolve())})
    logger.info("Se ha entrenado el modelo")
    return checkpoint


if __name__ == "__main__":
    main()
