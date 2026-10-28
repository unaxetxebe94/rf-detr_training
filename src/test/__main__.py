import argparse
import json
import sys
from pathlib import Path

import cv2

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.test"

from ..params import get_global_config
from ..preprocess.annotations import group_annotations_by_image, read_annotations
from .iou import xyxy_to_xywh
from .thresholds import find_optimal_thresholds
from .matching import match_predictions
from .drawing import render_image
from .metrics import detection_metrics
from .image_evaluation import evaluate_images, reconstruct_predictions


IOU_THRESHOLD = 0.3
RAW_THRESHOLD = 0.05
DEFAULT_THRESHOLD = 0.5
PRETRAIN_WEIGHTS = Path("trainings") / "training" / "checkpoint_best_total.pth"


def evaluate_dataset(model, test_dir: Path, output_dir: Path) -> dict:
    test_dir, output_dir = Path(test_dir), Path(output_dir)
    coco = read_annotations(test_dir / "_annotations.coco.json")
    reconstruct_predictions(coco, {})
    category_names = {cat["id"]: cat["name"] for cat in coco["categories"]}
    annotations = group_annotations_by_image(coco)
    ground_truth = {
        image["id"]: [
            (ann["bbox"], ann["category_id"])
            for ann in annotations.get(image["id"], [])
        ]
        for image in coco["images"]
    }
    for folder in ("TP", "FP", "FN", "misclassified"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    raw_predictions = {}
    for image in coco["images"]:
        image_path = test_dir / image["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        prediction = model.predict(str(image_path), threshold=RAW_THRESHOLD)
        raw_predictions[image["id"]] = [
            (xyxy_to_xywh(box), int(label), float(score))
            for box, label, score in zip(
                prediction.xyxy, prediction.class_id, prediction.confidence
            )
        ]
        print(f"{image['file_name']}: {len(raw_predictions[image['id']])} detecciones.")

    thresholds = find_optimal_thresholds(
        raw_predictions, ground_truth, category_names, iou_threshold=IOU_THRESHOLD
    )
    filtered_predictions = {
        image_id: [
            (box, label, score) for box, label, score in predictions
            if score >= thresholds.get(label, DEFAULT_THRESHOLD)
        ]
        for image_id, predictions in raw_predictions.items()
    }
    all_predictions = []
    for image in coco["images"]:
        image_id = image["id"]
        preds, gts = filtered_predictions[image_id], ground_truth[image_id]
        all_predictions.extend((image_id, *pred) for pred in preds)
        pred_boxes, pred_labels, pred_scores = (
            [pred[idx] for pred in preds] for idx in range(3)
        )
        gt_boxes, gt_labels = [gt[0] for gt in gts], [gt[1] for gt in gts]
        tp, fp, fn, mis = match_predictions(
            pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, IOU_THRESHOLD
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

    results = detection_metrics(all_predictions, ground_truth, IOU_THRESHOLD)
    results["per_class_thresholds"] = {
        category_names.get(label, str(label)): threshold
        for label, threshold in sorted(thresholds.items())
    }
    results["image_metrics"], results["image_detection_metrics"] = evaluate_images(
        coco, filtered_predictions, test_dir, output_dir / "doors", IOU_THRESHOLD
    )
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluacion por tile y por puerta.")
    parser.add_argument("--weights", type=Path, default=PRETRAIN_WEIGHTS)
    args = parser.parse_args(argv)
    cfg = get_global_config()
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall, RFDETRNano

    models = {
        "nano": RFDETRNano, "small": RFDETRSmall,
        "medium": RFDETRMedium, "large": RFDETRLarge,
    }
    model = models[cfg.model_type](pretrain_weights=str(args.weights))
    evaluate_dataset(model, cfg.final_data / "test", cfg.final_data / "test_output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
