"""Persistence of evaluation results in MLflow, metrics.json and per_class_metrics.csv."""

import csv
import json
import re
import warnings
from pathlib import Path

import mlflow

from ..pipelines.disk.mlflow import ensure_test_run, log_dict_metrics, log_dict_params

# Metric names shared by the three model families on the central MLflow server.
STANDARD_NAMES = {
    "Precision": "precision", "Recall": "recall", "F1-Score": "f1", "Accuracy": "accuracy",
    "Specificity": "specificity", "Total TP": "tp", "Total FP": "fp", "Total FN": "fn", "Total TN": "tn",
    "AUROC": "auroc", "AP": "ap", "mAP@50": "map50", "mAP@50:95": "map50_95",
    "mAR@50": "mar50", "mAR@50:95": "mar50_95",
}
PER_CLASS_CSV = "per_class_metrics.csv"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _standard(metrics: dict, prefix: str) -> dict:
    return {
        f"{prefix}/{STANDARD_NAMES[key]}": value for key, value in metrics.items()
        if key in STANDARD_NAMES and isinstance(value, (int, float))
    }


def standard_metrics(results: dict, split: str = "test") -> dict:
    """``{split}/{tile|image}/[class/<class>/][detection/]<metric>`` keys.

    Without ``detection/``: defective-vs-normal per tile or per door (overall = any class, per class =
    one-vs-rest). With ``detection/``: instance-level detection metrics (IoU-matched boxes).
    """
    detection_overall = {
        "tile": {key: value for key, value in results.items() if isinstance(value, (int, float))},
        "image": results.get("image_detection_metrics", {}),
    }
    metrics = {}
    for level in ("tile", "image"):
        base = f"{split}/{level}"
        for group, values in results.get(f"{level}_presence", {}).items():
            metrics.update(_standard(values, base if group == "overall" else f"{base}/class/{slug(group)}"))
        if f"{level}_presence" in results:
            metrics.update(_standard(detection_overall[level], f"{base}/detection"))
        for name, values in results.get(f"{level}_detection_per_class", {}).items():
            metrics.update(_standard(values, f"{base}/class/{slug(name)}/detection"))
            threshold = results.get("per_class_thresholds", {}).get(name)
            if isinstance(threshold, (int, float)):
                metrics[f"{base}/class/{slug(name)}/threshold"] = threshold
    return metrics


def log_results_to_mlflow(results: dict) -> str:
    run_id = ensure_test_run()
    # Legacy keys (tile/*, image/binary/*, image/detection/*) are kept so old runs stay comparable.
    tile_metrics = {
        key: value for key, value in results.items()
        if isinstance(value, (int, float))
    }
    log_dict_metrics(tile_metrics, prefix="tile/")
    log_dict_params({
        key: results[key] for key in ("evaluation_scope", "door_evaluation_skip_reason")
        if key in results
    })
    if "image_metrics" in results:
        log_dict_metrics(results["image_metrics"], prefix="image/binary/")
    if "image_detection_metrics" in results:
        log_dict_metrics(results["image_detection_metrics"], prefix="image/detection/")
    log_dict_metrics(standard_metrics(results))
    if "threshold_calibration" in results:
        log_dict_params(results["threshold_calibration"], prefix="calibration/")
    if "per_class_thresholds" in results:
        log_dict_params(results["per_class_thresholds"], prefix="threshold/")
    if "category_id_to_name" in results:
        log_dict_params({"category_id_to_name": results["category_id_to_name"]})
    return run_id


def per_class_rows(results: dict) -> list[dict]:
    rows = []
    for level in ("tile", "image"):
        presence = results.get(f"{level}_presence", {})
        detection = results.get(f"{level}_detection_per_class", {})
        for group in presence:
            row = {"level": level, "class": group}
            row.update({STANDARD_NAMES[k]: v for k, v in presence[group].items() if k in STANDARD_NAMES})
            row.update({
                f"detection_{STANDARD_NAMES[k]}": v
                for k, v in detection.get(group, {}).items() if k in STANDARD_NAMES
            })
            rows.append(row)
    return rows


def save_per_class_csv(results: dict, output_dir) -> Path | None:
    rows = per_class_rows(results)
    if not rows:
        return None
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(output_dir) / PER_CLASS_CSV
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def log_artifacts_to_mlflow(output_dir) -> None:
    """Upload the small evaluation summaries (not the rendered images) to the active run."""
    ensure_test_run()
    for name in ("metrics.json", PER_CLASS_CSV):
        path = Path(output_dir) / name
        if path.is_file():
            try:
                mlflow.log_artifact(str(path), artifact_path="evaluation")
            except Exception as error:  # a tracking failure must not lose the evaluation
                warnings.warn(f"No se pudo subir {path} a MLflow: {error}", stacklevel=2)


def save_results(results: dict, output_dir) -> None:
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
