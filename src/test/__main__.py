import argparse
import json
import sys
from math import isclose
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import cv2

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.test"

from ..core.params import get_data_config, get_global_config
from ..preprocess.annotations import group_annotations_by_image, read_annotations
from .drawing import render_image
from .image_evaluation import evaluate_images, reconstruct_predictions
from .iou import xyxy_to_xywh
from .matching import match_predictions
from .metrics import detection_metrics
from .thresholds import select_thresholds

IOU_THRESHOLD = 0.3
RAW_THRESHOLD = 0.05
DEFAULT_THRESHOLD = 0.5
PRETRAIN_WEIGHTS = Path("trainings") / "training" / "checkpoint_best_total.pth"


def _align_door_categories(coco, original_data_dirs):
    test_ids = {cat["name"]: cat["id"] for cat in coco["categories"]}
    if len(test_ids) != len(coco["categories"]):
        raise ValueError("Test category names must be unique for door alignment.")
    originals_by_source, originals_by_name = {}, {}
    for directory in dict.fromkeys(Path(path) for path in original_data_dirs):
        original = read_annotations(directory / "_annotations.coco.json")
        categories = {cat["id"]: cat["name"] for cat in original["categories"]}
        annotations = group_annotations_by_image(original)
        for image in original["images"]:
            source_id = uuid5(
                NAMESPACE_URL, str((directory / image["file_name"]).resolve())
            ).hex
            entry = (categories, image, {
                ann["id"]: ann for ann in annotations.get(image["id"], [])
            })
            originals_by_source[source_id] = entry
            originals_by_name.setdefault(image["file_name"], []).append(entry)

    sources = []
    for source in coco["source_images"]:
        entry = originals_by_source.get(source["id"])
        if entry is None:
            candidates = originals_by_name.get(source["original_file_name"], [])
            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot identify original dataset for door {source['original_file_name']}."
                )
            entry = candidates[0]
        categories, original_image, original_annotations = entry
        annotations = []
        for ann in source["annotations"]:
            original_ann = original_annotations.get(ann.get("id"))
            if original_ann is None:
                names = set()
                if original_image.get("width") and original_image.get("height"):
                    scale_x = source["width"] / original_image["width"]
                    scale_y = source["height"] / original_image["height"]
                    # Resize rounding can shift the scaled box by up to one pixel.
                    for candidate in original_annotations.values():
                        box = candidate.get("bbox")
                        if box is not None and all(
                            isclose(value, original_value * scale, abs_tol=1.0)
                            for value, original_value, scale in zip(
                                ann["bbox"], box, (scale_x, scale_y, scale_x, scale_y), strict=True
                            )
                        ):
                            names.add(categories[candidate["category_id"]])
                if len(names) != 1:
                    raise ValueError(
                        f"Cannot uniquely align annotation {ann.get('id')} for "
                        f"door {source['original_file_name']} by original ID or resized box."
                    )
                name = names.pop()
            else:
                name = categories[original_ann["category_id"]]
            if name in test_ids:
                annotations.append({**ann, "category_id": test_ids[name]})
        sources.append({
            **source, "annotations": annotations, "has_defects": bool(annotations),
        })
    return {**coco, "source_images": sources}


def _align_categories(coco, training_categories):
    training_ids = {cat["name"]: cat["id"] for cat in training_categories}
    if len(training_ids) != len(training_categories):
        raise ValueError("Training category names must be unique for alignment.")
    mapping = {
        cat["id"]: training_ids[cat["name"]]
        for cat in coco["categories"] if cat["name"] in training_ids
    }

    def align_annotations(annotations):
        return [
            {**ann, "category_id": mapping[ann["category_id"]]}
            for ann in annotations if ann["category_id"] in mapping
        ]

    aligned = {
        **coco,
        "categories": training_categories,
        "annotations": align_annotations(coco["annotations"]),
    }
    if "source_images" in coco:
        aligned["source_images"] = []
        for source in coco["source_images"]:
            annotations = align_annotations(source["annotations"])
            aligned["source_images"].append({
                **source, "annotations": annotations,
                "has_defects": bool(annotations),
            })
    removed = [cat["name"] for cat in coco["categories"] if cat["id"] not in mapping]
    print(f"Category alignment to training IDs: {mapping}; removed classes: {removed}")
    return aligned


def _ground_truth(coco):
    annotations = group_annotations_by_image(coco)
    return {
        image["id"]: [
            (ann["bbox"], ann["category_id"])
            for ann in annotations.get(image["id"], [])
        ]
        for image in coco["images"]
    }


def _predict_dataset(model, directory, coco, raw_threshold=RAW_THRESHOLD):
    allowed_labels = {cat["id"] for cat in coco["categories"]}
    raw_predictions = {}
    for image in coco["images"]:
        image_path = directory / image["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        prediction = model.predict(str(image_path), threshold=raw_threshold)
        raw_predictions[image["id"]] = [
            (xyxy_to_xywh(box), int(label) + 1, float(score))
            for box, label, score in zip(
                prediction.xyxy, prediction.class_id, prediction.confidence, strict=True
            )
            if int(label) + 1 in allowed_labels
        ]
        print(f"{image['file_name']}: {len(raw_predictions[image['id']])} detecciones.")
    return raw_predictions


def evaluate_dataset(model, test_dir: Path, output_dir: Path,
                     valid_dir: Path | None = None,
                     global_threshold: float | None = None,
                     train_annotations: Path | None = None,
                     original_data_dirs: tuple[Path, Path] | None = None) -> dict:
    if global_threshold is not None and not 0 <= global_threshold <= 1:
        raise ValueError("Global threshold must be between 0 and 1.")
    test_dir, output_dir = Path(test_dir), Path(output_dir)
    coco = read_annotations(test_dir / "_annotations.coco.json")
    reconstruct_predictions(coco, {})
    training = read_annotations(
        Path(train_annotations) if train_annotations is not None
        else test_dir.parent / "train" / "_annotations.coco.json"
    )
    checkpoint_names = getattr(getattr(model, "model", None), "class_names", None)
    if isinstance(checkpoint_names, (list, tuple)) and checkpoint_names:
        training_names = [
            cat["name"] for cat in training["categories"]
            if cat.get("supercategory") != "none"
        ]
        if list(checkpoint_names) != training_names:
            raise ValueError(
                "Checkpoint classes/order do not match training annotations. "
                "Use --train-annotations with the original training COCO JSON. "
                f"Checkpoint: {list(checkpoint_names)}; dataset: {training_names}"
            )
    if original_data_dirs is not None:
        coco = _align_door_categories(coco, original_data_dirs)
    coco = _align_categories(coco, training["categories"])
    category_names = {cat["id"]: cat["name"] for cat in coco["categories"]}
    if global_threshold is None:
        valid_dir = Path(valid_dir) if valid_dir is not None else test_dir.parent / "valid"
        if valid_dir.resolve() == test_dir.resolve():
            raise ValueError("Validation and test must use separate directories.")
        validation = read_annotations(valid_dir / "_annotations.coco.json")
        validation = _align_categories(validation, training["categories"])
        validation_names = {cat["id"]: cat["name"] for cat in validation["categories"]}
        validation_gt = _ground_truth(validation)
        if not any(validation_gt.values()):
            raise ValueError("Validation needs ground-truth annotations to calibrate thresholds.")
        print(f"\nCalibrating thresholds on validation: {valid_dir}")
        thresholds, calibration = select_thresholds(
            _predict_dataset(model, valid_dir, validation), validation_gt,
            validation_names, iou_threshold=IOU_THRESHOLD,
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
    ground_truth = _ground_truth(coco)
    raw_predictions = _predict_dataset(
        model, test_dir, coco,
        raw_threshold=0.0 if global_threshold is not None else RAW_THRESHOLD,
    )
    for folder in ("TP", "FP", "FN", "misclassified"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    default_threshold = (
        calibration["global_threshold"]
        if calibration["selected_strategy"] == "global" else DEFAULT_THRESHOLD
    )
    filtered_predictions = {
        image_id: [
            (box, label, score) for box, label, score in predictions
            if score >= thresholds.get(label, default_threshold)
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
    results["category_id_to_name"] = {
        str(label): name for label, name in category_names.items()
    }
    results["threshold_calibration"] = calibration
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


def _confidence_threshold(value):
    try:
        threshold = float(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError("Threshold must be a float between 0 and 1.") from err
    if not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError("Threshold must be between 0 and 1.")
    return threshold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluacion por tile y por puerta.")
    parser.add_argument("--weights", type=Path, default=PRETRAIN_WEIGHTS)
    parser.add_argument(
        "--train-annotations", type=Path,
        help="Original training COCO JSON defining the checkpoint's class IDs.",
    )
    parser.add_argument(
        "--global-threshold", type=_confidence_threshold, default=None,
        help="Use this confidence threshold for all classes, skipping validation calibration.",
    )
    args = parser.parse_args(argv)
    cfg = get_global_config()
    data_cfg = get_data_config()
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    models = {
        "nano": RFDETRNano, "small": RFDETRSmall,
        "medium": RFDETRMedium, "large": RFDETRLarge,
    }
    model = models[cfg.model_type](pretrain_weights=str(args.weights))
    train_annotations = args.train_annotations
    snapshot = args.weights.parent / "training_categories.json"
    if train_annotations is None and snapshot.is_file():
        train_annotations = snapshot
    evaluate_dataset(
        model, data_cfg.training_dataset_path / "test",
        data_cfg.training_dataset_path / "test_output",
        global_threshold=args.global_threshold,
        train_annotations=train_annotations,
        original_data_dirs=tuple(data_cfg.sources),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
