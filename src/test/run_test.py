"""Command-line entry point for checkpoint evaluation."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.test"

from ..pipelines.disk.mlflow import ensure_test_run, log_dict_params, mlflow_run
from ..pipelines.disk.params import get_data_config, get_global_config
from .evaluation import evaluate_dataset


def _confidence_threshold(value):
    try:
        threshold = float(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError("Threshold must be a float between 0 and 1.") from err
    if not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError("Threshold must be between 0 and 1.")
    return threshold


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Evaluacion por tile y por puerta.")
    parser.add_argument(
        "--weights", type=Path,
        help="Checkpoint to evaluate; defaults to the latest dated training directory.",
    )
    parser.add_argument(
        "--train-annotations", type=Path,
        help="Original training COCO JSON defining the checkpoint's class IDs.",
    )
    parser.add_argument(
        "--global-threshold", type=_confidence_threshold, default=None,
        help="Use this confidence threshold for all classes, skipping validation calibration.",
    )
    return parser.parse_args(argv)


def get_latest_model_weights() -> str:
    training_dir = Path("trainings")
    latest_date = datetime(1, 1, 1)
    latest_model = None
    if not training_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")
    for directory in training_dir.iterdir():
        if not directory.is_dir():
            continue
        try:
            date = datetime.strptime(directory.name[:19], "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            print(f"Skipping directory without a date-formatted name: {directory.name}")
            continue
        if date > latest_date:
            latest_date = date
            latest_model = directory / "checkpoint_best_total.pth"
    if latest_model is None or not latest_model.is_file():
        raise FileNotFoundError(
            f"Best-total checkpoint not found in the latest training directory: {latest_model}"
        )
    return str(latest_model)


def run_test(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    weights = args.weights if args.weights is not None else Path(get_latest_model_weights())
    if not weights.is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {weights}")
    with mlflow_run("test"):
        cfg = get_global_config()
        data_cfg = get_data_config()
        run_id = ensure_test_run()
        print(f"Run de MLflow para el test: {run_id}")
        from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

        models = {
            "nano": RFDETRNano, "small": RFDETRSmall,
            "medium": RFDETRMedium, "large": RFDETRLarge,
        }
        model = models[cfg.model_type](pretrain_weights=str(weights))
        train_annotations = args.train_annotations
        snapshot = weights.parent / "training_categories.json"
        if train_annotations is None and snapshot.is_file():
            train_annotations = snapshot
        log_dict_params({"test/checkpoint": str(weights.resolve())})
        log_dict_params({"test/dataset_dir": str(data_cfg.training_dataset_path.resolve())})
        evaluate_dataset(
            model, data_cfg.training_dataset_path / "test",
            data_cfg.training_dataset_path / "test_output",
            global_threshold=args.global_threshold,
            train_annotations=train_annotations,
            original_data_dirs=tuple(data_cfg.sources),
        )
    return 0
