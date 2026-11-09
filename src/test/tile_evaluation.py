"""Detection metrics and annotated output images for individual tiles."""

from pathlib import Path

import cv2

from ..preprocess.annotations import group_annotations_by_image
from .drawing import render_image
from .matching import match_predictions
from .metrics import detection_metrics


def ground_truth_by_image(coco):
    annotations = group_annotations_by_image(coco)
    return {
        image["id"]: [
            (ann["bbox"], ann["category_id"])
            for ann in annotations.get(image["id"], [])
        ]
        for image in coco["images"]
    }


def evaluate_tiles(coco, predictions_by_tile, test_dir, output_dir, iou_threshold=0.3):
    test_dir, output_dir = Path(test_dir), Path(output_dir)
    category_names = {cat["id"]: cat["name"] for cat in coco["categories"]}
    ground_truth = ground_truth_by_image(coco)
    for folder in ("TP", "FP", "FN", "misclassified"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    all_predictions = []
    for image in coco["images"]:
        image_id = image["id"]
        preds, gts = predictions_by_tile[image_id], ground_truth[image_id]
        all_predictions.extend((image_id, *pred) for pred in preds)
        pred_boxes, pred_labels, pred_scores = (
            [pred[idx] for pred in preds] for idx in range(3)
        )
        gt_boxes, gt_labels = [gt[0] for gt in gts], [gt[1] for gt in gts]
        tp, fp, fn, mis = match_predictions(
            pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold
        )
        rendered = render_image(
            test_dir / image["file_name"], gt_boxes, gt_labels, category_names,
            pred_boxes, pred_labels, pred_scores, category_names, tp, fp, fn, mis,
        )
        buckets = []
        if len(tp) == len(gts) and not (fp or fn or mis):
            buckets.append("TP")
        if fp:
            buckets.append("FP")
        if fn:
            buckets.append("FN")
        if mis:
            buckets.append("misclassified")
        for bucket in buckets:
            destination = output_dir / bucket / image["file_name"]
            if not cv2.imwrite(str(destination), rendered):
                raise OSError(f"No se pudo guardar el tile: {destination}")
    return detection_metrics(all_predictions, ground_truth, iou_threshold)
