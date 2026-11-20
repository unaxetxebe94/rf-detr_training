"""Coordinate dataset preparation, threshold calibration and evaluation."""

from pathlib import Path

from ..preprocess.annotations import read_annotations
from .categories import align_categories, align_door_categories, validate_checkpoint_categories
from .image_evaluation import (
    door_evaluation_unavailable,
    evaluate_images,
    predictions_by_source,
    reconstruct_predictions,
    source_ground_truth,
)
from .metrics import per_class_detection_metrics, presence_metrics
from .predictions import DEFAULT_THRESHOLD, RAW_THRESHOLD, filter_predictions, predict_dataset
from .reporting import log_artifacts_to_mlflow, log_results_to_mlflow, save_per_class_csv, save_results
from .thresholds import select_thresholds
from .tile_evaluation import evaluate_tiles, ground_truth_by_image

IOU_THRESHOLD = 0.3


def _flatten(predictions_by_image):
    return [
        (image_id, *prediction)
        for image_id, predictions in predictions_by_image.items()
        for prediction in predictions
    ]


def class_level_metrics(coco, filtered_predictions, raw_predictions, category_names, doors: bool) -> dict:
    """Defective-vs-normal (overall and per class) and per-class detection metrics per tile and per door."""
    tile_truth = ground_truth_by_image(coco)
    results = {
        "tile_presence": presence_metrics(tile_truth, filtered_predictions, category_names, raw_predictions),
        "tile_detection_per_class": per_class_detection_metrics(
            _flatten(filtered_predictions), tile_truth, category_names, IOU_THRESHOLD,
        ),
    }
    if doors:
        door_truth = source_ground_truth(coco)
        door_predictions = reconstruct_predictions(coco, filtered_predictions)
        results["image_presence"] = presence_metrics(
            door_truth, door_predictions, category_names, predictions_by_source(coco, raw_predictions),
        )
        results["image_detection_per_class"] = per_class_detection_metrics(
            _flatten(door_predictions), door_truth, category_names, IOU_THRESHOLD,
        )
    return results


def _calibrate_thresholds(model, test_dir, valid_dir, training_categories):
    valid_dir = Path(valid_dir) if valid_dir is not None else test_dir.parent / "valid"
    if valid_dir.resolve() == test_dir.resolve():
        raise ValueError("Validation and test must use separate directories.")
    validation = read_annotations(valid_dir / "_annotations.coco.json")
    validation = align_categories(validation, training_categories)
    category_names = {cat["id"]: cat["name"] for cat in validation["categories"]}
    ground_truth = ground_truth_by_image(validation)
    if not any(ground_truth.values()):
        raise ValueError("Validation needs ground-truth annotations to calibrate thresholds.")
    print(f"\nCalibrating thresholds on validation: {valid_dir}")
    return select_thresholds(
        predict_dataset(model, valid_dir, validation), ground_truth,
        category_names, iou_threshold=IOU_THRESHOLD,
    )


def evaluate_dataset(model, test_dir: Path, output_dir: Path,
                     valid_dir: Path | None = None,
                     global_threshold: float | None = None,
                     train_annotations: Path | None = None,
                     original_data_dirs: tuple[Path, ...] | None = None) -> dict:
    if global_threshold is not None and not 0 <= global_threshold <= 1:
        raise ValueError("Global threshold must be between 0 and 1.")
    test_dir, output_dir = Path(test_dir), Path(output_dir)
    coco = read_annotations(test_dir / "_annotations.coco.json")
    door_skip_reason = door_evaluation_unavailable(coco, test_dir)
    if door_skip_reason is None:
        reconstruct_predictions(coco, {})
    else:
        print(
            f"AVISO: {door_skip_reason} Se evaluaran solo los tiles. "
            "Una carpeta doors de ejecuciones anteriores no corresponde a este test."
        )
        coco = {**coco, "source_images": []}
    training = read_annotations(
        Path(train_annotations) if train_annotations is not None
        else test_dir.parent / "train" / "_annotations.coco.json"
    )
    validate_checkpoint_categories(model, training["categories"])
    if original_data_dirs is not None and door_skip_reason is None:
        coco = align_door_categories(coco, original_data_dirs)
    coco = align_categories(coco, training["categories"])
    category_names = {cat["id"]: cat["name"] for cat in coco["categories"]}

    if global_threshold is None:
        thresholds, calibration = _calibrate_thresholds(
            model, test_dir, valid_dir, training["categories"],
        )
    else:
        thresholds = dict.fromkeys(category_names, global_threshold)
        calibration = {
            "split": None,
            "source": "manual",
            "global_threshold": global_threshold,
            "selected_strategy": "global",
        }
    print(f"\nEvaluating test with fixed {calibration['selected_strategy']} thresholds: {test_dir}")
    raw_predictions = predict_dataset(
        model, test_dir, coco,
        raw_threshold=0.0 if global_threshold is not None else RAW_THRESHOLD,
    )
    default_threshold = (
        calibration["global_threshold"]
        if calibration["selected_strategy"] == "global" else DEFAULT_THRESHOLD
    )
    filtered_predictions = filter_predictions(raw_predictions, thresholds, default_threshold)
    results = evaluate_tiles(coco, filtered_predictions, test_dir, output_dir, IOU_THRESHOLD)
    results["category_id_to_name"] = {
        str(label): name for label, name in category_names.items()
    }
    results["threshold_calibration"] = calibration
    results["per_class_thresholds"] = {
        category_names.get(label, str(label)): threshold
        for label, threshold in sorted(thresholds.items())
    }
    results["evaluation_scope"] = "tiles" if door_skip_reason else "tiles_and_doors"
    if door_skip_reason is None:
        results["image_metrics"], results["image_detection_metrics"] = evaluate_images(
            coco, filtered_predictions, test_dir, output_dir / "doors", IOU_THRESHOLD
        )
    else:
        results["door_evaluation_skip_reason"] = door_skip_reason
    results.update(class_level_metrics(
        coco, filtered_predictions, raw_predictions, category_names, doors=door_skip_reason is None,
    ))
    log_results_to_mlflow(results)
    save_results(results, output_dir)
    save_per_class_csv(results, output_dir)
    log_artifacts_to_mlflow(output_dir)
    return results
