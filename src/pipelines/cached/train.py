"""Training on the cached view: incremental tile cache on disk + DataLoaders with online augmentation.

Usage: ``uv run python -m src.pipelines.cached [--full] [--skip-update] [--deep-verify] [--resume RUN] [--no-test]``.
"""

import argparse
import datetime
import json
import logging
from dataclasses import asdict
from pathlib import Path

from rfdetr.training import RFDETRModelModule, build_trainer

from ..disk.logger import get_logger
from ..disk.mlflow import get_mlflow_logger, log_dict_params, mlflow_run
from ..disk.params import (
    get_cached_config,
    get_data_config,
    get_global_config,
    get_preprocess_config,
    get_train_config,
)
from ..disk.seed import set_seed
from ..disk.train import MODEL_CONFIGS, get_rfdetr_train_cfg
from .datamodule import CachedDataModule, augmentation_settings
from .prepare import add_prepare_arguments, locks, prepare


def resolve_resume(resume: Path | None) -> tuple[Path | None, Path | None]:
    """(checkpoint, output_dir) for ``--resume``: a training directory (uses its ``last.ckpt``) or a ``.ckpt`` file."""
    if resume is None:
        return None, None
    resume = Path(resume)
    checkpoint = resume / "last.ckpt" if resume.is_dir() else resume
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No existe el checkpoint para reanudar: {checkpoint}")
    return checkpoint, checkpoint.parent


def rfdetr_overrides(cached_cfg, checkpoint: Path | None) -> dict:
    """``TrainConfig`` fields set from ``cached:`` (``None`` keeps RF-DETR's own default)."""
    backend, aug_config, _ = augmentation_settings(cached_cfg)
    update = {
        "augmentation_backend": backend,
        "aug_config": aug_config,
        "num_workers": cached_cfg.num_workers,
        "amp_dtype": cached_cfg.amp_dtype,
        "eval_interval": cached_cfg.eval_interval,
        "early_stopping": cached_cfg.early_stopping,
        "early_stopping_patience": cached_cfg.early_stopping_patience,
        # The test split is evaluated by src.test.run_test after training (per tile and per door).
        "run_test": False,
        "resume": str(checkpoint) if checkpoint else None,
    }
    optional = {
        "pin_memory": cached_cfg.pin_memory,
        "persistent_workers": cached_cfg.persistent_workers,
        "prefetch_factor": cached_cfg.prefetch_factor,
        "eval_batch_size": cached_cfg.eval_batch_size,
    }
    update.update({key: value for key, value in optional.items() if value is not None})
    return update


def _check_backend(cached_cfg) -> None:
    """Fail before updating the cache when the Kornia (GPU) backend cannot run."""
    if cached_cfg.augmentation_backend == "kornia":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("cached.augmentation-backend=kornia necesita una GPU CUDA.")


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Pipeline cached: caché de tiles incremental + entrenamiento.")
    add_prepare_arguments(parser)
    parser.add_argument("--resume", type=Path, default=None,
                        help="Directorio de un entrenamiento (usa su last.ckpt) o checkpoint .ckpt a reanudar.")
    parser.add_argument("--no-test", action="store_true", help="No ejecuta run_test al terminar.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directorio de salida del entrenamiento.")
    args = parser.parse_args(argv)
    if args.full and args.skip_update:
        parser.error("--full y --skip-update son incompatibles.")

    # The locks stay held while training and testing so nobody rewrites the tiles being read.
    with mlflow_run("pipeline"), locks(get_data_config(), get_cached_config()):
        checkpoint = train(args)
        if not args.no_test:
            from ...test.run_test import run_test

            status = run_test(["--weights", str(checkpoint)])
            if status:
                raise RuntimeError(f"run_test ha terminado con código {status}.")
    return checkpoint


def train(args: argparse.Namespace) -> Path:
    """Prepare the view and train; the caller must hold :func:`locks`."""
    logger = get_logger(__name__, level=logging.DEBUG)
    global_cfg = get_global_config()
    preprocess_cfg = get_preprocess_config()
    train_cfg = get_train_config()
    cached_cfg = get_cached_config()
    _check_backend(cached_cfg)
    set_seed(global_cfg.seed)

    resume_checkpoint, resume_dir = resolve_resume(args.resume)
    output_dir = Path(
        args.output_dir or resume_dir or Path("trainings", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow_run("train"):
        view, prepare_stats = prepare(full=args.full, skip_update=args.skip_update, deep_verify=args.deep_verify)
        empty = [split for split in ("train", "valid") if not view.summary.get(f"{split}/tiles")]
        if empty:
            raise ValueError(f"Splits sin tiles: {empty}; revisa los ratios y las fuentes.")
        with open(output_dir / "training_categories.json", "w", encoding="utf-8") as f:
            json.dump({"categories": view.categories}, f, indent=2)

        model_config = MODEL_CONFIGS[global_cfg.model_type](num_classes=len(view.categories))
        rfdetr_train_cfg = get_rfdetr_train_cfg(view.dataset_dir, output_dir, train_cfg).model_copy(
            update=rfdetr_overrides(cached_cfg, resume_checkpoint)
        )
        if resume_checkpoint:
            from rfdetr.detr import _resolve_resumed_optimizer_settings

            rfdetr_train_cfg = _resolve_resumed_optimizer_settings(rfdetr_train_cfg)

        log_dict_params(asdict(global_cfg), prefix="global/")
        log_dict_params(asdict(preprocess_cfg), prefix="preprocess/")
        log_dict_params({key: str(value) for key, value in asdict(cached_cfg).items()}, prefix="cached/")
        log_dict_params({**view.summary, "fingerprint": view.fingerprint, **prepare_stats}, prefix="dataset/")
        log_dict_params(rfdetr_train_cfg.model_dump(mode="json"), prefix="train/")
        log_dict_params({"training/output_dir": str(output_dir.resolve())})

        _, _, augment_prob = augmentation_settings(cached_cfg)
        trainer = build_trainer(rfdetr_train_cfg, model_config)
        trainer.loggers = [*trainer.loggers, get_mlflow_logger()]
        module = RFDETRModelModule(model_config, rfdetr_train_cfg)
        data_module = CachedDataModule(
            model_config, rfdetr_train_cfg, view,
            negative_ratio=cached_cfg.negative_ratio,
            augment_prob=augment_prob,
            max_transforms_per_sample=preprocess_cfg.max_transforms_per_sample,
            max_bad_samples=cached_cfg.max_bad_samples,
            seed=global_cfg.seed,
        )
        trainer.fit(module, data_module, ckpt_path=str(resume_checkpoint) if resume_checkpoint else None)

    checkpoint = output_dir / "checkpoint_best_total.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"El entrenamiento no ha generado el checkpoint: {checkpoint}")
    log_dict_params({"training/checkpoint": str(checkpoint.resolve())})
    logger.info("Se ha entrenado el modelo")
    return checkpoint
