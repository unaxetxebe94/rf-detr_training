"""Persistence of evaluation results in MLflow and metrics.json."""

import json

from ..pipelines.disk.mlflow import ensure_test_run, log_dict_metrics, log_dict_params


def log_results_to_mlflow(results: dict) -> str:
    run_id = ensure_test_run()
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
    if "threshold_calibration" in results:
        log_dict_params(results["threshold_calibration"], prefix="calibration/")
    if "per_class_thresholds" in results:
        log_dict_params(results["per_class_thresholds"], prefix="threshold/")
    if "category_id_to_name" in results:
        log_dict_params({"category_id_to_name": results["category_id_to_name"]})
    return run_id


def save_results(results: dict, output_dir) -> None:
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
