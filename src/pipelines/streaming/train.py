"""Training with tiles generated in memory by the DataLoaders (no dataset written to disk).

Usage: ``uv run python -m src.streaming``. Same params.yaml as the disk pipeline: the sources are
split per original image, the tiles of each original are cropped on the fly from a single decode,
and train copies (``augmentations-per-image``) get random photometric augmentation online.
"""

import datetime
import json
import logging
from dataclasses import asdict
from pathlib import Path

from rfdetr.training import RFDETRModelModule, build_trainer

from ..disk.logger import get_logger
from ..disk.mlflow import get_mlflow_logger, log_dict_params, mlflow_run
from ..disk.params import (
    get_data_config,
    get_global_config,
    get_preprocess_config,
    get_streaming_config,
    get_train_config,
)
from ..disk.seed import set_seed
from ..disk.train import MODEL_CONFIGS, get_rfdetr_train_cfg
from .datamodule import StreamingDataModule
from .plan import build_tile_plan


def main(output_dir: Path | None = None) -> Path:
    with mlflow_run("train"):
        return _train(output_dir)


def _train(output_dir: Path | None) -> Path:
    logger = get_logger(__name__, level=logging.DEBUG)
    global_cfg = get_global_config()
    data_cfg = get_data_config()
    preprocess_cfg = get_preprocess_config()
    train_cfg = get_train_config()
    streaming_cfg = get_streaming_config()
    set_seed(global_cfg.seed)

    output_dir = Path(output_dir or Path("trainings", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = build_tile_plan(data_cfg.sources, preprocess_cfg, global_cfg)
    logger.info(f"Plan de tiles en memoria: {plan.summary()}")
    with open(output_dir / "training_categories.json", "w", encoding="utf-8") as f:
        json.dump({"categories": plan.categories}, f, indent=2)

    model_config = MODEL_CONFIGS[global_cfg.model_type](num_classes=len(plan.categories))
    # dataset_dir is mandatory for RF-DETR but unused: the datasets are injected by StreamingDataModule.
    rfdetr_train_cfg = get_rfdetr_train_cfg(output_dir, output_dir, train_cfg).model_copy(
        update={"num_workers": streaming_cfg.num_workers, "run_test": True}
    )

    log_dict_params(asdict(global_cfg), prefix="global/")
    log_dict_params(asdict(preprocess_cfg), prefix="preprocess/")
    log_dict_params(asdict(streaming_cfg), prefix="streaming/")
    log_dict_params(plan.summary(), prefix="dataset/")
    log_dict_params(rfdetr_train_cfg.model_dump(mode="json"), prefix="train/")
    log_dict_params({"training/output_dir": str(output_dir.resolve())})

    trainer = build_trainer(rfdetr_train_cfg, model_config)
    trainer.loggers = [*trainer.loggers, get_mlflow_logger()]
    module = RFDETRModelModule(model_config, rfdetr_train_cfg)
    data_module = StreamingDataModule(
        model_config, rfdetr_train_cfg, plan,
        augmentations_per_image=preprocess_cfg.augmentations_per_image,
        max_transforms_per_sample=preprocess_cfg.max_transforms_per_sample,
        images_per_worker=streaming_cfg.images_per_worker,
        seed=global_cfg.seed,
    )

    trainer.fit(module, data_module)
    checkpoint = output_dir / "checkpoint_best_total.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"El entrenamiento no ha generado el checkpoint: {checkpoint}")
    log_dict_params({"training/checkpoint": str(checkpoint.resolve())})
    logger.info("Se ha entrenado el modelo")
    return checkpoint
