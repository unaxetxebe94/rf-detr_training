"""FP/FN counts when uniformly shifting the calibrated class thresholds."""

import math

from .image_evaluation import door_bucket
from .matching import match_predictions
from .predictions import filter_predictions


def _count_errors(predictions, ground_truth, iou_threshold):
    counts = dict.fromkeys(("TP", "FP", "FN"), 0)
    for image_id, gts in ground_truth.items():
        preds = predictions.get(image_id, [])
        tp, fp, fn, misclassified = match_predictions(
            [pred[0] for pred in preds], [pred[1] for pred in preds],
            [pred[2] for pred in preds], [gt[0] for gt in gts],
            [gt[1] for gt in gts], iou_threshold,
        )
        counts["TP"] += len(tp)
        counts["FP"] += len(fp) + len(misclassified)
        counts["FN"] += len(fn) + len(misclassified)
    return counts


def _count_image_decisions(predictions, ground_truth):
    counts = dict.fromkeys(("TP", "FP", "FN", "TN"), 0)
    for image_id, gts in ground_truth.items():
        counts[door_bucket(bool(gts), bool(predictions.get(image_id, [])))] += 1
    return counts


def sweep_confidence_thresholds(predictions, ground_truth, baseline_thresholds,
                               step=0.01, iou_threshold=0.3, binary=False):
    if not baseline_thresholds or any(
        not math.isfinite(value) or not 0 <= value <= 1
        for value in baseline_thresholds.values()
    ):
        raise ValueError("Baseline thresholds must be nonempty and between 0 and 1.")
    if not math.isfinite(step) or not 0 < step <= 1:
        raise ValueError("Step must be greater than 0 and at most 1.")
    labels = (
        {label for gts in ground_truth.values() for _, label in gts}
        | {label for preds in predictions.values() for _, label, _ in preds}
    )
    if not labels <= baseline_thresholds.keys():
        raise ValueError("Every ground-truth and prediction class needs a baseline threshold.")
    if predictions.keys() - ground_truth.keys():
        raise ValueError("Predictions contain image IDs not present in the evaluated dataset.")

    lower = round(-min(baseline_thresholds.values()), 10)
    upper = round(1 - max(baseline_thresholds.values()), 10)
    offsets = {lower, 0.0, upper}
    for direction, distance in ((-1, -lower), (1, upper)):
        offsets.update(
            round(direction * index * step, 10)
            for index in range(1, math.ceil(distance / step))
        )

    rows = []
    for offset in sorted(offsets):
        thresholds = {
            label: min(1.0, max(0.0, round(value + offset, 10)))
            for label, value in baseline_thresholds.items()
        }
        filtered = filter_predictions(predictions, thresholds)
        counts = (
            _count_image_decisions(filtered, ground_truth) if binary
            else _count_errors(filtered, ground_truth, iou_threshold)
        )
        rows.append({"offset": offset, "thresholds": thresholds, **counts})
    return rows
