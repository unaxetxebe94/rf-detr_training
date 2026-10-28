from collections import defaultdict
from pathlib import Path

import cv2

from .drawing import render_image
from .iou import compute_iou
from .matching import match_predictions
from .metrics import binary_metrics, detection_metrics


def door_bucket(has_defects: bool, has_predictions: bool) -> str:
    if has_defects:
        return "TP" if has_predictions else "FN"
    return "FP" if has_predictions else "TN"


def suppress_overlapping_predictions(predictions, iou_threshold=0.5):
    kept = []
    for prediction in sorted(predictions, key=lambda pred: -pred[2]):
        box, label, _ = prediction
        if not any(
            label == other_label and compute_iou(box, other_box) > iou_threshold
            for other_box, other_label, _ in kept
        ):
            kept.append(prediction)
    return kept


def reconstruct_predictions(coco, predictions_by_tile):
    sources = {source["id"]: source for source in coco.get("source_images", [])}
    if not sources:
        raise ValueError(
            "El dataset no contiene imagenes completas. Vuelve a ejecutar el preprocesado."
        )
    grouped = defaultdict(list)
    for tile in coco["images"]:
        source_id = tile.get("source_image_id")
        if source_id not in sources or "tile_origin" not in tile:
            raise ValueError(f"Falta el origen del tile {tile['file_name']}.")
        offset_x, offset_y = tile["tile_origin"]
        for box, label, score in predictions_by_tile.get(tile["id"], []):
            x, y, width, height = box
            grouped[source_id].append((
                [x + offset_x, y + offset_y, width, height], label, score,
            ))
    return {
        source_id: suppress_overlapping_predictions(grouped[source_id])
        for source_id in sources
    }


def evaluate_images(coco, predictions_by_tile, test_dir, output_dir, iou_threshold=0.3):
    predictions = reconstruct_predictions(coco, predictions_by_tile)
    counts = dict.fromkeys(("TP", "FP", "FN", "TN"), 0)
    labels = {cat["id"]: cat["name"] for cat in coco["categories"]}
    all_predictions, ground_truth = [], {}
    output_dir = Path(output_dir)
    for bucket in counts:
        (output_dir / bucket).mkdir(parents=True, exist_ok=True)

    for source in coco["source_images"]:
        source_id = source["id"]
        preds = predictions[source_id]
        gts = [(ann["bbox"], ann["category_id"]) for ann in source["annotations"]]
        ground_truth[source_id] = gts
        all_predictions.extend((source_id, *pred) for pred in preds)
        bucket = door_bucket(source["has_defects"], bool(preds))
        counts[bucket] += 1
        image_path = Path(test_dir) / source["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Imagen completa no encontrada: {image_path}")
        pred_boxes, pred_labels, pred_scores = (
            [pred[idx] for pred in preds] for idx in range(3)
        )
        gt_boxes, gt_labels = [gt[0] for gt in gts], [gt[1] for gt in gts]
        tp, fp, fn, mis = match_predictions(
            pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold,
        )
        rendered = render_image(
            image_path, gt_boxes, gt_labels, labels,
            pred_boxes, pred_labels, pred_scores, labels, tp, fp, fn, mis,
        )
        destination = output_dir / bucket / Path(source["file_name"]).name
        if not cv2.imwrite(str(destination), rendered):
            raise OSError(f"No se pudo guardar la imagen: {destination}")
        print(f"Puerta {source['original_file_name']}: {bucket}")
    return (
        binary_metrics(counts),
        detection_metrics(all_predictions, ground_truth, iou_threshold),
    )
