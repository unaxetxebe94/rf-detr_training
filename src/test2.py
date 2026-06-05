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

IOU_THRESHOLDS_MAP = np.arange(0.5, 1.0, 0.05)   # for mAP@50:95

with open("test2_params.yaml", mode="r") as f:
    params = yaml.safe_load(f)
IOU_THRESHOLD = params["iou-threshold"]
RAW_THRESHOLD = params["raw-threshold"]
DEFAULT_THRESHOLD = params["default-threshold"]
PRETRAIN_WEIGHTS = params["pretrain-weights"]
OUTPUT_DIR = params["output-dir"]


# ──────────────────────────────────────────────
# IoU helpers
# ──────────────────────────────────────────────

def compute_iou(box_a, box_b):
    """Compute IoU between two boxes in [x, y, w, h] format."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def box_xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def xyxy_to_xywh(b):
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


# ──────────────────────────────────────────────
# Match predictions → ground-truths for one image
# ──────────────────────────────────────────────

def match_predictions(pred_boxes, pred_labels, pred_scores,
                      gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Returns:
        tp_indices   - set of pred indices that are true positives
        fp_indices   - set of pred indices that are false positives
        fn_indices   - set of gt indices that were missed
        mis_indices  - set of pred indices matched geometrically but wrong class
    """
    matched_gt = set()
    tp_pred, fp_pred, mis_pred = set(), set(), set()
    fn_gt = set(range(len(gt_boxes)))

    # Sort predictions by descending confidence
    order = np.argsort(pred_scores)[::-1]

    for pi in order:
        best_iou, best_gi = 0.0, -1
        for gi, gt_box in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(pred_boxes[pi], gt_box)
            if iou > best_iou:
                best_iou, best_gi = iou, gi

        if best_iou >= iou_threshold and best_gi != -1:
            matched_gt.add(best_gi)
            fn_gt.discard(best_gi)
            if pred_labels[pi] == gt_labels[best_gi]:
                tp_pred.add(pi)
            else:
                mis_pred.add(pi)          # geometrically matched, wrong class
        else:
            fp_pred.add(pi)

    return tp_pred, fp_pred, fn_gt, mis_pred


# ──────────────────────────────────────────────
# AP computation (VOC-style interpolation)
# ──────────────────────────────────────────────

def compute_ap(recalls, precisions):
    """11-point interpolated AP."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t] if np.any(recalls >= t) else np.array([0.0])
        ap += np.max(p) / 11
    return ap


def compute_map(all_results, gt_by_image, iou_threshold=0.5):
    """
    all_results : list of (img_id, pred_box, pred_label, pred_score)
    gt_by_image : dict img_id → list of (gt_box, gt_label)
    """
    # Group by class
    by_class = defaultdict(list)
    for img_id, box, label, score in all_results:
        by_class[label].append((img_id, box, score))

    gt_by_class = defaultdict(lambda: defaultdict(list))
    for img_id, pairs in gt_by_image.items():
        for box, label in pairs:
            gt_by_class[label][img_id].append(box)

    aps = []
    for cls, preds in by_class.items():
        preds_sorted = sorted(preds, key=lambda x: -x[2])
        detected = defaultdict(set)
        tp_list, fp_list = [], []

        for img_id, box, score in preds_sorted:
            gt_boxes = gt_by_class[cls][img_id]
            best_iou, best_gi = 0.0, -1
            for gi, gt_box in enumerate(gt_boxes):
                if gi in detected[img_id]:
                    continue
                iou = compute_iou(box, gt_box)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi

            if best_iou >= iou_threshold and best_gi != -1:
                detected[img_id].add(best_gi)
                tp_list.append(1)
                fp_list.append(0)
            else:
                tp_list.append(0)
                fp_list.append(1)

        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum(fp_list)
        n_gt = sum(len(v) for v in gt_by_class[cls].values())
        recalls = tp_cum / (n_gt + 1e-9)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
        aps.append(compute_ap(recalls, precisions))

    return np.mean(aps) if aps else 0.0


# ──────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────

COLORS = {
    "gt":           (0, 188, 0),    
    "tp":           (0, 255, 0),  
    "fp":           (0, 137, 255),
    "fn":           (0, 0, 255),
    "mis":          (0, 255, 255),
}


def draw_box(img, box, label, color, thickness=2):
    x1, y1, x2, y2 = box_xywh_to_xyxy(box)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def render_image(img_path, gt_boxes, gt_labels, gt_label_names,
                 pred_boxes, pred_labels, pred_scores, pred_label_names,
                 tp_pred, fp_pred, fn_gt, mis_pred):
    """
    Returns a single side-by-side image:
      left  = ground truth (green = normal GT, orange = missed GT)
      right = predictions  (cyan = TP, red = FP, magenta = misclassified)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)

    left  = img.copy()
    right = img.copy()

    # --- Ground-truth side ---
    for gi, (box, lbl) in enumerate(zip(gt_boxes, gt_labels)):
        color = COLORS["fn"] if gi in fn_gt else COLORS["gt"]
        tag   = f"GT-MISS:{gt_label_names.get(lbl, lbl)}" if gi in fn_gt \
                else f"GT:{gt_label_names.get(lbl, lbl)}"
        draw_box(left, box, tag, color)

    # --- Prediction side ---
    for pi, (box, lbl, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        if pi in tp_pred:
            color, prefix = COLORS["tp"], "TP"
        elif pi in mis_pred:
            color, prefix = COLORS["mis"], "MIS"
        else:
            color, prefix = COLORS["fp"], "FP"
        tag = f"{prefix}:{pred_label_names.get(lbl, lbl)} {score:.2f}"
        draw_box(right, box, tag, color)

    combined = np.concatenate([left, right], axis=1)
    return combined


# ──────────────────────────────────────────────
# Per-class optimal threshold search
# ──────────────────────────────────────────────

def find_optimal_thresholds(raw_preds_by_img, gt_by_img, category_id_to_name,
                             iou_threshold=0.5,
                             threshold_range=None):
    """
    Sweep confidence thresholds per class and return the one that maximises F1.

    raw_preds_by_img : dict  img_id -> [(box_xywh, class_id, score), ...]
    gt_by_img        : dict  img_id -> [(box_xywh, class_id), ...]
    Returns          : dict  class_id -> best_threshold (float)
    """
    if threshold_range is None:
        threshold_range = np.arange(0.05, 1.0, 0.05)

    all_classes = {lbl for pairs in gt_by_img.values() for _, lbl in pairs}

    best_thresholds = {}
    print("\nSearching optimal confidence threshold per class …")

    for cls in sorted(all_classes):
        best_f1 = -1.0
        best_thresh = 0.5

        for thresh in threshold_range:
            tp = fp = fn = 0

            for img_id, gt_pairs in gt_by_img.items():
                gt_boxes_cls = [b for b, l in gt_pairs if l == cls]
                preds_for_img = raw_preds_by_img.get(img_id, [])
                pred_boxes_cls = [b for b, l, s in preds_for_img
                                  if l == cls and s >= thresh]

                # Greedy match (sorted by order already at collection time)
                matched_gt = set()
                for pb in pred_boxes_cls:
                    best_iou, best_gi = 0.0, -1
                    for gi, gb in enumerate(gt_boxes_cls):
                        if gi in matched_gt:
                            continue
                        iou = compute_iou(pb, gb)
                        if iou > best_iou:
                            best_iou, best_gi = iou, gi
                    if best_iou >= iou_threshold and best_gi != -1:
                        matched_gt.add(best_gi)
                        tp += 1
                    else:
                        fp += 1
                fn += len(gt_boxes_cls) - len(matched_gt)

            precision = tp / (tp + fp + 1e-9)
            recall    = tp / (tp + fn + 1e-9)
            f1        = 2 * precision * recall / (precision + recall + 1e-9)

            if f1 > best_f1:
                best_f1    = f1
                best_thresh = thresh

        best_thresholds[cls] = float(round(best_thresh, 4))
        cls_name = category_id_to_name.get(cls, str(cls))
        print(f"  {cls_name:<25} thresh={best_thresh:.2f}  F1={best_f1:.4f}")

    return best_thresholds


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    out_base = OUTPUT_DIR
    with open("params.yaml", mode="r") as f:
        params = yaml.safe_load(f)

    # ── Model ──────────────────────────────────
    model = RFDETRLarge(pretrain_weights=PRETRAIN_WEIGHTS)

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
    out_base = Path("test_results")
    for folder in ("TP", "FP", "FN", "misclassified"):
        (out_base / folder).mkdir(parents=True, exist_ok=True)

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
            cv2.imwrite(str(out_base / "TP" / test_img_name), combined)
        if has_fp:
            cv2.imwrite(str(out_base / "FP" / test_img_name), combined)
        if has_fn:
            cv2.imwrite(str(out_base / "FN" / test_img_name), combined)
        if has_mis:
            cv2.imwrite(str(out_base / "misclassified" / test_img_name), combined)

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

    metrics_path = out_base / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Annotated images saved to {out_base}/")