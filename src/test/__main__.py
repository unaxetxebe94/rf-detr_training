import os
import json
import yaml
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall, RFDETRNano
from utils import read_params
import cv2

from test.iou import xyxy_to_xywh, compute_iou
from test.thresholds import find_optimal_thresholds
from test.matching import match_predictions
from test.drawing import render_image
from test.metrics import compute_map


IOU_THRESHOLD      = 0.3
RAW_THRESHOLD      = 0.05   # collect all candidates; per-class thresholds applied later
IOU_THRESHOLDS_MAP = np.arange(0.5, 1.0, 0.05)   # for mAP@50:95
DEFAULT_THRESHOLD  = 0.5    # fallback for classes not seen in GT
PRETRAIN_WEIGHTS = r"E:\rf-detr_training\trainings\training\checkpoint_best_total.pth"


MODEL_SIZE_TO_CLASS_MAP = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge
}


if __name__ == "__main__":

    with open("params.yaml", mode="r") as f:
        params = yaml.safe_load(f)

    OUTPUT_DIR = Path(params["final-data"]) / "test_output"

    # ── Model ──────────────────────────────────
    model = MODEL_SIZE_TO_CLASS_MAP[params["model-size"]](pretrain_weights=PRETRAIN_WEIGHTS)

    # ── Dataset ────────────────────────────────
    test_dir = Path(params["final-data"], r"test")
    with open(test_dir / "_annotations.coco.json", mode="r") as f:
        anns = json.load(f)

    # Build lookup structures
    id_to_filename = {img["id"]: img["file_name"] for img in anns["images"]}
    filename_to_id = {v: k for k, v in id_to_filename.items()}
    category_id_to_name = {cat["id"]: cat["name"] for cat in anns["categories"]}

    gt_by_img: dict[int, list] = defaultdict(list)   # img_id → [(box, cat_id), …]
    for ann in anns["annotations"]:
        gt_by_img[ann["image_id"]].append((ann["bbox"], ann["category_id"]))

    # ── Output folders ─────────────────────────
    for folder in ("TP", "FP", "FN", "misclassified"):
        (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)

    test_img_names = [
        p for p in os.listdir(test_dir)
        if p.lower().endswith((".tiff", ".jpg", ".png"))
    ]

    # ══════════════════════════════════════════════
    # Phase 1: Collect raw predictions (low threshold)
    # ══════════════════════════════════════════════
    print(f"\n{'='*45}")
    print("  PHASE 1 — Collecting raw predictions")
    print(f"{'='*45}")

    raw_preds_by_img = {}   # img_id → [(box_xywh, label, score), …]
    gt_eval = {}            # img_id → [(box_xywh, label), …]

    for i, test_img_name in enumerate(test_img_names, 1):
        test_img_path = test_dir / test_img_name
        img_id = filename_to_id.get(test_img_name)

        gts = gt_by_img.get(img_id, [])
        gt_eval[img_id] = [(g[0], g[1]) for g in gts]

        prediction = model.predict(str(test_img_path), threshold=RAW_THRESHOLD)

        pred_boxes_xyxy = [list(b) for b in prediction.xyxy]
        pred_labels     = list(prediction.class_id)
        pred_scores     = list(prediction.confidence)
        pred_boxes_xywh = [xyxy_to_xywh(b) for b in pred_boxes_xyxy]

        raw_preds_by_img[img_id] = list(zip(pred_boxes_xywh, pred_labels, pred_scores))
        print(f"  [{i}/{len(test_img_names)}] {test_img_name}  ({len(pred_boxes_xywh)} raw detections)")

    # ══════════════════════════════════════════════
    # Phase 2: Find optimal confidence threshold per class
    # ══════════════════════════════════════════════
    print(f"\n{'='*45}")
    print("  PHASE 2 — Threshold optimisation")
    print(f"{'='*45}")

    per_class_thresholds = find_optimal_thresholds(
        raw_preds_by_img, gt_eval, category_id_to_name,
        iou_threshold=IOU_THRESHOLD
    )

    print(f"\n{'='*45}")
    print("  OPTIMAL CONFIDENCE THRESHOLD PER CLASS")
    print(f"{'='*45}")
    for cls_id, thresh in sorted(per_class_thresholds.items()):
        cls_name = category_id_to_name.get(cls_id, str(cls_id))
        print(f"  {cls_name:<25} {thresh:.2f}")
    print(f"{'='*45}\n")

    # ══════════════════════════════════════════════
    # Phase 3: Final evaluation with per-class thresholds
    # ══════════════════════════════════════════════
    print(f"{'='*45}")
    print("  PHASE 3 — Final evaluation")
    print(f"{'='*45}")

    all_tp = all_fp = all_fn = 0
    all_preds_for_map = []   # (img_id, box, label, score)
    gt_for_map = {}          # img_id → [(box, label), …]

    for test_img_name in test_img_names:
        test_img_path = test_dir / test_img_name
        img_id = filename_to_id.get(test_img_name)

        # ── Ground truth for this image ─────────
        gts           = gt_by_img.get(img_id, [])
        gt_boxes_img  = [g[0] for g in gts]
        gt_labels_img = [g[1] for g in gts]
        gt_for_map[img_id] = [(b, l) for b, l in zip(gt_boxes_img, gt_labels_img)]

        # ── Apply per-class thresholds ───────────
        raw_preds = raw_preds_by_img.get(img_id, [])
        filtered  = [
            (b, l, s) for b, l, s in raw_preds
            if s >= per_class_thresholds.get(l, DEFAULT_THRESHOLD)
        ]
        pred_boxes_xywh = [p[0] for p in filtered]
        pred_labels_img = [p[1] for p in filtered]
        pred_scores_img = [p[2] for p in filtered]

        for b, l, s in filtered:
            all_preds_for_map.append((img_id, b, l, s))

        # ── Match ───────────────────────────────
        tp_pred, fp_pred, fn_gt_set, mis_pred = match_predictions(
            pred_boxes_xywh, pred_labels_img, pred_scores_img,
            gt_boxes_img, gt_labels_img, iou_threshold=IOU_THRESHOLD
        )

        n_tp = len(tp_pred)
        n_fp = len(fp_pred) + len(mis_pred)
        n_fn = len(fn_gt_set)

        all_tp += n_tp
        all_fp += n_fp
        all_fn += n_fn

        # ── Render combined image ────────────────
        combined = render_image(
            test_img_path,
            gt_boxes_img, gt_labels_img, category_id_to_name,
            pred_boxes_xywh, pred_labels_img, pred_scores_img, category_id_to_name,
            tp_pred, fp_pred, fn_gt_set, mis_pred
        )

        # ── Decide output buckets ────────────────
        is_perfect = (n_tp == len(gt_boxes_img)) and (n_fp == 0) and (n_fn == 0) and (len(mis_pred) == 0)
        has_fp     = len(fp_pred) > 0
        has_fn     = len(fn_gt_set) > 0
        has_mis    = len(mis_pred) > 0

        if is_perfect:
            cv2.imwrite(str(OUTPUT_DIR / "TP" / test_img_name), combined)
        if has_fp:
            cv2.imwrite(str(OUTPUT_DIR / "FP" / test_img_name), combined)
        if has_fn:
            cv2.imwrite(str(OUTPUT_DIR / "FN" / test_img_name), combined)
        if has_mis:
            cv2.imwrite(str(OUTPUT_DIR / "misclassified" / test_img_name), combined)

        print(f"[{test_img_name}]  TP={n_tp}  FP={len(fp_pred)}  FN={n_fn}  MIS={len(mis_pred)}")

    # ──────────────────────────────────────────
    # Global metrics
    # ──────────────────────────────────────────
    precision = all_tp / (all_tp + all_fp + 1e-9)
    recall    = all_tp / (all_tp + all_fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    # mAP@50
    map50 = compute_map(all_preds_for_map, gt_for_map, iou_threshold=0.50)

    # mAP@50:95
    aps_multi = [
        compute_map(all_preds_for_map, gt_for_map, iou_threshold=t)
        for t in IOU_THRESHOLDS_MAP
    ]
    map50_95 = float(np.mean(aps_multi))

    # mAR@50
    def compute_mar(all_results, gt_by_image, iou_threshold=0.5, max_dets=100):
        recalls_per_img = []
        preds_by_img = defaultdict(list)
        for img_id, box, label, score in all_results:
            preds_by_img[img_id].append((box, label, score))

        for img_id, gt_pairs in gt_by_image.items():
            gt_boxes  = [p[0] for p in gt_pairs]
            gt_labels = [p[1] for p in gt_pairs]
            preds = sorted(preds_by_img[img_id], key=lambda x: -x[2])[:max_dets]
            if not gt_boxes:
                continue
            pred_b = [p[0] for p in preds]
            pred_l = [p[1] for p in preds]
            pred_s = [p[2] for p in preds]
            tp_, _, fn_, mis_ = match_predictions(pred_b, pred_l, pred_s,
                                                   gt_boxes, gt_labels, iou_threshold)
            recalls_per_img.append(len(tp_) / (len(gt_boxes) + 1e-9))

        return float(np.mean(recalls_per_img)) if recalls_per_img else 0.0

    mar50    = compute_mar(all_preds_for_map, gt_for_map, iou_threshold=0.50)
    mar50_95 = float(np.mean([
        compute_mar(all_preds_for_map, gt_for_map, iou_threshold=t)
        for t in IOU_THRESHOLDS_MAP
    ]))

    # ──────────────────────────────────────────
    # Report
    # ──────────────────────────────────────────
    results = {
        "mAP@50":       round(map50,     4),
        "mAP@50:95":    round(map50_95,  4),
        "mAR@50":       round(mar50,     4),
        "mAR@50:95":    round(mar50_95,  4),
        "Precision":    round(precision, 4),
        "Recall":       round(recall,    4),
        "F1-Score":     round(f1,        4),
        "Total TP":     all_tp,
        "Total FP":     all_fp,
        "Total FN":     all_fn,
    }

    print("\n" + "="*45)
    print("          EVALUATION RESULTS")
    print("="*45)
    for k, v in results.items():
        print(f"  {k:<18} {v}")
    print("="*45)

    # Include per-class thresholds in the saved JSON
    results["per_class_thresholds"] = {
        category_id_to_name.get(cls_id, str(cls_id)): thresh
        for cls_id, thresh in sorted(per_class_thresholds.items())
    }

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Annotated images saved to {OUTPUT_DIR}/")