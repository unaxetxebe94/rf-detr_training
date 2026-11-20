"""Mirror RF-DETR validation metrics to the keys shared by every model family on the MLflow server."""

import re
import warnings

import mlflow
from pytorch_lightning import Callback

# RF-DETR validation metrics are instance-level detection metrics over tiles.
VAL_KEYS = {
    "val/mAP_50": "map50",
    "val/mAP_50_95": "map50_95",
    "val/mAP_75": "map75",
    "val/mAR": "mar",
    "val/F1": "f1",
    "val/precision": "precision",
    "val/recall": "recall",
}
CLASS_AP_PREFIX = "val/AP/"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def standard_val_metrics(callback_metrics) -> dict[str, float]:
    metrics = {}
    for key, value in callback_metrics.items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN
            continue
        if key in VAL_KEYS:
            metrics[f"val/tile/detection/{VAL_KEYS[key]}"] = value
        elif key.startswith(CLASS_AP_PREFIX):
            metrics[f"val/tile/class/{_slug(key[len(CLASS_AP_PREFIX):])}/detection/map50_95"] = value
    return metrics


class StandardValMetrics(Callback):
    """Log ``val/tile/...`` keys (step = epoch) into the active MLflow run after each validation."""

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or mlflow.active_run() is None:
            return
        metrics = standard_val_metrics(trainer.callback_metrics)
        if not metrics:
            return
        try:
            mlflow.log_metrics(metrics, step=trainer.current_epoch)
        except Exception as error:  # tracking must never stop a training
            warnings.warn(f"No se pudieron registrar las métricas de validación en MLflow: {error}", stacklevel=2)
