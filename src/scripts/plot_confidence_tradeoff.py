"""Export a customer-facing FP/FN chart around validation-optimal thresholds."""

import argparse
import csv
import json
import math
import sys
import textwrap
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.scripts"

from ..core.params import get_data_config, get_global_config
from ..preprocess.annotations import read_annotations
from ..test.categories import align_categories, align_door_categories, validate_checkpoint_categories
from ..test.confidence_sweep import sweep_confidence_thresholds
from ..test.evaluation import IOU_THRESHOLD
from ..test.image_evaluation import door_evaluation_unavailable, reconstruct_predictions
from ..test.predictions import predict_dataset
from ..test.run_test import get_latest_model_weights
from ..test.thresholds import select_thresholds
from ..test.tile_evaluation import ground_truth_by_image


def build_report(model, test_dir, valid_dir, train_annotations, step=0.01):
    return build_reports(model, test_dir, valid_dir, train_annotations, step, level="tiles")["tiles"]


def build_reports(model, test_dir, valid_dir, train_annotations, step=0.01,
                  level="all", original_data_dirs=None):
    if level not in ("tiles", "images", "all"):
        raise ValueError("Level must be tiles, images or all.")
    test_dir, valid_dir = Path(test_dir), Path(valid_dir)
    if test_dir.resolve() == valid_dir.resolve():
        raise ValueError("Validation and test must use separate directories.")
    training = read_annotations(Path(train_annotations))
    validate_checkpoint_categories(model, training["categories"])
    categories = [cat for cat in training["categories"] if cat.get("supercategory") != "none"]
    category_names = {cat["id"]: cat["name"] for cat in categories}
    if not category_names or len(category_names) != len(categories):
        raise ValueError("Training must define defect categories with unique IDs.")
    test = read_annotations(test_dir / "_annotations.coco.json")
    image_skip_reason = None
    if level != "tiles":
        image_skip_reason = door_evaluation_unavailable(test, test_dir, require_image_files=False)
        if image_skip_reason:
            if level == "images":
                raise ValueError(f"Original-image curves unavailable: {image_skip_reason}")
            print(f"WARNING: Original-image curves skipped: {image_skip_reason}")
        else:
            reconstruct_predictions(test, {})
            if original_data_dirs:
                test = align_door_categories(test, original_data_dirs)
    datasets = {}
    for split, directory in (("valid", valid_dir), ("test", test_dir)):
        coco = test if split == "test" else read_annotations(directory / "_annotations.coco.json")
        if split == "valid" or level == "tiles" or image_skip_reason:
            coco.pop("source_images", None)
        datasets[split] = align_categories(coco, categories)
    validation_gt = ground_truth_by_image(datasets["valid"])
    if not any(validation_gt.values()):
        raise ValueError("Validation needs ground-truth annotations to calibrate thresholds.")
    ground_truth = ground_truth_by_image(datasets["test"])
    if not ground_truth:
        raise ValueError("Test needs at least one tile.")

    # Keep low-confidence predictions: the sweep must include thresholds below 0.05.
    baseline, calibration = select_thresholds(
        predict_dataset(model, valid_dir, datasets["valid"], raw_threshold=0.0),
        validation_gt, category_names, iou_threshold=IOU_THRESHOLD,
    )
    predictions = predict_dataset(model, test_dir, datasets["test"], raw_threshold=0.0)
    common = {
        "calibration_split": "valid",
        "calibration_unit": "tile_detection",
        "evaluation_split": "test",
        "test_dir": str(test_dir.resolve()),
        "valid_dir": str(valid_dir.resolve()),
        "train_annotations": str(Path(train_annotations).resolve()),
        "iou_threshold": IOU_THRESHOLD,
        "tile_count": len(ground_truth),
        "category_id_to_name": category_names,
        "baseline_thresholds": baseline,
        "calibration": calibration,
        "step": step,
    }
    reports = {}
    if level in ("tiles", "all"):
        reports["tiles"] = {
            **common, "evaluation_unit": "tile_detection",
            "ground_truth_defects": sum(len(gts) for gts in ground_truth.values()),
            "rows": sweep_confidence_thresholds(predictions, ground_truth, baseline, step, IOU_THRESHOLD),
        }
        if image_skip_reason:
            reports["tiles"]["original_image_skip_reason"] = image_skip_reason
    if level != "tiles" and image_skip_reason is None:
        image_coco = datasets["test"]
        image_ground_truth = {
            source["id"]: [(ann["bbox"], ann["category_id"]) for ann in source["annotations"]]
            for source in image_coco["source_images"]
        }
        # Within each class, NMS always keeps the higher score. Applying a shared
        # class threshold afterwards is equivalent to filtering before NMS.
        image_predictions = reconstruct_predictions(image_coco, predictions)
        image_common = {
            **common,
            "original_image_count": len(image_ground_truth),
            "defective_image_count": sum(bool(gts) for gts in image_ground_truth.values()),
            "ground_truth_defects": sum(len(gts) for gts in image_ground_truth.values()),
            "nms_iou_threshold": 0.5,
        }
        reports["original_image_detection"] = {
            **image_common, "evaluation_unit": "original_image_detection",
            "rows": sweep_confidence_thresholds(
                image_predictions, image_ground_truth, baseline, step, IOU_THRESHOLD,
            ),
        }
        reports["original_image_binary"] = {
            **image_common, "evaluation_unit": "original_image_binary", "iou_threshold": None,
            "decision_rule": "Reject an image if any defect prediction passes its class threshold.",
            "rows": sweep_confidence_thresholds(
                image_predictions, image_ground_truth, baseline, step, IOU_THRESHOLD, binary=True,
            ),
        }
    return reports


def _plot_report(report, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    rows = report["rows"]
    base = next(row for row in rows if row["offset"] == 0)
    nearby = [row for row in rows if abs(row["offset"]) <= 0.1]
    unit = report["evaluation_unit"]
    binary = unit == "original_image_binary"
    titles = {
        "tile_detection": "Tile detections: false alarms vs missed defects",
        "original_image_detection": "Original-image detections: false alarms vs missed defects",
        "original_image_binary": "Original-image decisions: false rejects vs missed defective images",
    }
    fp_label = "FP: defect-free images rejected" if binary else "FP: incorrect or duplicate detections"
    fn_label = "FN: defective images accepted" if binary else "FN: missed defects"
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    try:
        fig.suptitle(titles[unit], fontsize=15)
        for ax, selected, title in (
            (axes[0], rows, "Full sweep: left = more permissive / right = stricter"),
            (axes[1], nearby, "Detail around the baseline (+/-10 percentage points)"),
        ):
            x = [row["offset"] * 100 for row in selected]
            for key, color, label in (
                ("FP", "#0072B2", fp_label),
                ("FN", "#D55E00", fn_label),
            ):
                ax.plot(x, [row[key] for row in selected], color=color, linewidth=2.5, label=label)
                ax.scatter([0], [base[key]], color=color, s=40, zorder=4)
                if ax is axes[1]:
                    ax.annotate(f"{key}={base[key]}", (0, base[key]), xytext=(8, 8),
                                textcoords="offset points", color=color, weight="bold")
            ax.axvline(0, color="#555555", linestyle="--", label="Baseline: best validation tile F1")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Change applied to all thresholds (percentage points; 0 = baseline)")
            ax.set_ylabel("Original images in test" if binary else "Detection errors in test")
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis="y", alpha=0.25)
            ax.legend(loc="best", fontsize=9)
        thresholds = " | ".join(
            f"{report['category_id_to_name'][label]}: {value:.2f}"
            for label, value in report["baseline_thresholds"].items()
        )
        if unit == "tile_detection":
            sample = f"Test: {report['tile_count']} tiles, {report['ground_truth_defects']} annotated defects"
            interpretation = "Errors per tile, not unique original images. A wrong class counts as FP and FN."
        elif binary:
            total, defective = report["original_image_count"], report["defective_image_count"]
            sample = f"Test: {total} original images, {defective} defective, {total - defective} defect-free"
            interpretation = "Any retained defect prediction rejects an image. No class or IoU matching."
        else:
            sample = (
                f"Test: {report['original_image_count']} original images, "
                f"{report['ground_truth_defects']} annotated defects"
            )
            interpretation = "Reconstructed boxes; per-class NMS IoU=0.5. A wrong class counts as FP and FN."
        if not binary:
            sample += f" | Matching IoU={report['iou_threshold']}"
        footer = (
            f"Baseline: FP={base['FP']}, FN={base['FN']} | {sample}\n"
            + textwrap.fill(f"Baseline thresholds by class: {thresholds}", width=135)
            + f"\nEndpoints: first class reaches 0 or 1. {interpretation}"
            + "\nBaseline optimizes validation tile F1, not business cost. "
            "Choosing with these curves uses test as a decision set."
        )
        if "checkpoint" in report:
            footer += (
                f"\nModel: {report['model_type']} | "
                f"Training: {Path(report['checkpoint']).parent.name}"
            )
        fig.text(0.06, 0.02, footer, fontsize=9, va="bottom")
        fig.tight_layout(rect=(0.02, 0.17, 0.99, 0.96), h_pad=2.5)
        for extension in ("png", "svg"):
            fig.savefig(output_dir / f"confidence_tradeoff.{extension}", dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)


def save_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_report(report, output_dir)
    with open(output_dir / "confidence_tradeoff.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    labels = sorted(report["baseline_thresholds"])
    count_keys = ["TP", "FP", "FN"]
    if report["evaluation_unit"] == "original_image_binary":
        count_keys.append("TN")
    with open(output_dir / "confidence_tradeoff.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["offset", "offset_pp", *count_keys, *[f"threshold_{label}" for label in labels]])
        for row in report["rows"]:
            writer.writerow([
                row["offset"], round(row["offset"] * 100, 8), *[row[key] for key in count_keys],
                *[row["thresholds"][label] for label in labels],
            ])


def _step(value):
    step = float(value)
    if not math.isfinite(step) or not 0 < step <= 1:
        raise argparse.ArgumentTypeError("Step must be greater than 0 and at most 1.")
    return step


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="Defaults to the latest dated training checkpoint.")
    parser.add_argument("--dataset-dir", type=Path, help="Dataset root; defaults to params.yaml.")
    parser.add_argument("--train-annotations", type=Path, help="Override the checkpoint's training categories.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to DATASET/test_output/confidence_tradeoff.")
    parser.add_argument("--step", type=_step, default=0.01, help="Uniform threshold step (default: 0.01 = 1 pp).")
    parser.add_argument(
        "--level", choices=("tiles", "images", "all"), default="all",
        help="Default: all available levels. Images exports detection and binary decision curves.",
    )
    parser.add_argument(
        "--original-data-dir", type=Path, action="append",
        help="Original COCO dataset directory for image category alignment; repeat for multiple sources.",
    )
    args = parser.parse_args(argv)
    weights = args.weights if args.weights is not None else Path(get_latest_model_weights())
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    data_cfg = get_data_config()
    dataset_dir = args.dataset_dir or data_cfg.training_dataset_path
    output_dir = args.output_dir or dataset_dir / "test_output" / "confidence_tradeoff"
    train_annotations = args.train_annotations
    if train_annotations is None:
        snapshot = weights.parent / "training_categories.json"
        train_annotations = snapshot if snapshot.is_file() else dataset_dir / "train" / "_annotations.coco.json"
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    model_type = get_global_config().model_type
    models = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}
    model = models[model_type](pretrain_weights=str(weights))
    reports = build_reports(
        model, dataset_dir / "test", dataset_dir / "valid", train_annotations, args.step,
        args.level, args.original_data_dir if args.original_data_dir is not None else data_cfg.sources,
    )
    for level, report in reports.items():
        report.update(checkpoint=str(weights.resolve()), model_type=model_type)
        destination = output_dir if level == "tiles" else output_dir / level
        save_report(report, destination)
        base = next(row for row in report["rows"] if row["offset"] == 0)
        print(f"{level}: baseline FP={base['FP']}, FN={base['FN']}; {len(report['rows'])} points.")
        print(f"PNG/SVG charts and CSV/JSON data: {destination.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
