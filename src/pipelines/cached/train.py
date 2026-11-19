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

import mlflow

from ...preprocess.tile_creation import TILE_SIZE_MAPPER
from ..disk.logger import get_logger
from ..disk.mlflow import get_mlflow_logger, log_dict_params, mlflow_run, set_run_name
from ..disk.params import (
    get_cached_config,
    get_data_config,
    get_global_config,
    get_preprocess_config,
    get_train_config,
    params_path,
)
from ..disk.seed import set_seed
from ..disk.train import MODEL_CONFIGS, get_rfdetr_train_cfg
from .datamodule import CachedDataModule, augmentation_settings
from .prepare import add_prepare_arguments, locks, prepare
from .run_info import (
    dataset_report,
    environment_info,
    epoch_size,
    register_lineage,
    run_label,
    save_run_info,
)


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


def resolve_init(init_from: Path | None) -> Path | None:
    """Weights for ``--init-from``: a training directory (its best checkpoint) or a checkpoint file."""
    if init_from is None:
        return None
    init_from = Path(init_from)
    checkpoint = init_from / "checkpoint_best_total.pth" if init_from.is_dir() else init_from
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No existe el checkpoint para inicializar los pesos: {checkpoint}")
    return checkpoint


def check_init_categories(checkpoint: Path, categories: list[dict]) -> None:
    """A warm start is only meaningful when the class ids/names are the ones the checkpoint was trained with."""
    snapshot = checkpoint.parent / "training_categories.json"
    if not snapshot.is_file():
        logging.getLogger(__name__).warning(f"{snapshot} no existe: no se puede comprobar que las clases coincidan.")
        return
    with open(snapshot, encoding="utf-8") as f:
        previous = {(cat["id"], str(cat["name"])) for cat in json.load(f)["categories"]}
    if previous != {(cat["id"], str(cat["name"])) for cat in categories}:
        raise ValueError(
            f"Las clases actuales no coinciden con las del checkpoint inicial ({snapshot}); "
            "entrena desde cero o mantén las mismas clases."
        )


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
    parser.add_argument("--init-from", type=Path, default=None,
                        help="Entrenamiento incremental: directorio de un entrenamiento (usa su "
                             "checkpoint_best_total.pth) o checkpoint cuyos pesos inicializan el modelo. "
                             "Optimizador y epochs empiezan de cero.")
    parser.add_argument("--no-test", action="store_true", help="No ejecuta run_test al terminar.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directorio de salida del entrenamiento.")
    args = parser.parse_args(argv)
    if args.full and args.skip_update:
        parser.error("--full y --skip-update son incompatibles.")
    if args.resume and args.init_from:
        parser.error("--resume y --init-from son incompatibles.")

    # The locks stay held while training and testing so nobody rewrites the tiles being read.
    with mlflow_run("pipeline"), locks(get_data_config(), get_cached_config()):
        checkpoint = train(args)
        mlflow.set_tag("train_status", "finished")
        if args.no_test:
            mlflow.set_tag("test_status", "skipped")
        else:
            from ...test.run_test import run_test

            mlflow.set_tag("test_status", "running")
            try:
                status = run_test(["--weights", str(checkpoint)])
            except BaseException:
                mlflow.set_tag("test_status", "failed")
                raise
            if status:
                mlflow.set_tag("test_status", "failed")
                raise RuntimeError(f"run_test ha terminado con código {status}.")
            mlflow.set_tag("test_status", "finished")
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
    init_checkpoint = resolve_init(getattr(args, "init_from", None))
    output_dir = Path(
        args.output_dir or resume_dir or Path("trainings", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow_run("train"):
        view, prepare_stats = prepare(full=args.full, skip_update=args.skip_update, deep_verify=args.deep_verify)
        empty = [split for split in ("train", "valid") if not view.summary.get(f"{split}/tiles")]
        if empty:
            raise ValueError(f"Splits sin tiles: {empty}; revisa los ratios y las fuentes.")
        if init_checkpoint:
            check_init_categories(init_checkpoint, view.categories)
        with open(output_dir / "training_categories.json", "w", encoding="utf-8") as f:
            json.dump({"categories": view.categories}, f, indent=2)

        model_kwargs = {"pretrain_weights": str(init_checkpoint)} if init_checkpoint else {}
        model_config = MODEL_CONFIGS[global_cfg.model_type](num_classes=len(view.categories), **model_kwargs)
        rfdetr_train_cfg = get_rfdetr_train_cfg(view.dataset_dir, output_dir, train_cfg).model_copy(
            update=rfdetr_overrides(cached_cfg, resume_checkpoint)
        )
        if resume_checkpoint:
            from rfdetr.detr import _resolve_resumed_optimizer_settings

            rfdetr_train_cfg = _resolve_resumed_optimizer_settings(rfdetr_train_cfg)

        report, splits = dataset_report(view.dataset_dir, view.categories)
        content = {
            **view.summary, **report,
            "train/tiles_per_epoch": epoch_size(view.summary, cached_cfg.negative_ratio),
            "tile_size": TILE_SIZE_MAPPER[global_cfg.model_type],
            "tile_format": view.tile_format,
            "categories": [cat["name"] for cat in view.categories],
        }
        parameters = {**content, "fingerprint": view.fingerprint}
        deltas = register_lineage(init_checkpoint, parameters)
        set_run_name(run_label(parameters, init_checkpoint is not None))
        # saving-prob and augmentations-per-image belong to the disk pipeline and have no effect here.
        preprocess_params = {
            key: value for key, value in asdict(preprocess_cfg).items()
            if key not in ("saving_prob", "augmentations_per_image")
        }

        log_dict_params(asdict(global_cfg), prefix="global/")
        log_dict_params(preprocess_params, prefix="preprocess/")
        log_dict_params({key: str(value) for key, value in asdict(cached_cfg).items()}, prefix="cached/")
        log_dict_params({**parameters, **prepare_stats}, prefix="dataset/")
        log_dict_params(deltas, prefix="delta/")
        log_dict_params(rfdetr_train_cfg.model_dump(mode="json"), prefix="train/")
        log_dict_params(environment_info(), prefix="env/")
        log_dict_params({"training/output_dir": str(output_dir.resolve())})
        save_run_info(
            output_dir,
            {
                "dataset_report.json": {**parameters, **prepare_stats, "delta_vs_parent": deltas},
                "splits.json": splits,
                "environment.json": environment_info(),
                "rfdetr_train_config.json": rfdetr_train_cfg.model_dump(mode="json"),
            },
            extra_files=(params_path(),),
        )

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
