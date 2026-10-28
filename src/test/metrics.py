import numpy as np
from collections import defaultdict

from .iou import compute_iou
from .matching import match_predictions


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
    for cls in set(by_class) | set(gt_by_class):
        preds = by_class[cls]
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
        recalls = tp_cum / n_gt if n_gt else np.zeros_like(tp_cum, dtype=float)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        aps.append(compute_ap(recalls, precisions))

    return np.mean(aps) if aps else 0.0


def compute_mar(all_results, gt_by_image, iou_threshold=0.5, max_dets=100):
    recalls = []
    preds_by_img = defaultdict(list)
    for img_id, box, label, score in all_results:
        preds_by_img[img_id].append((box, label, score))
    for img_id, gt_pairs in gt_by_image.items():
        if not gt_pairs:
            continue
        preds = sorted(preds_by_img[img_id], key=lambda pred: -pred[2])[:max_dets]
        tp, _, _, _ = match_predictions(
            [pred[0] for pred in preds], [pred[1] for pred in preds],
            [pred[2] for pred in preds], [gt[0] for gt in gt_pairs],
            [gt[1] for gt in gt_pairs], iou_threshold,
        )
        recalls.append(len(tp) / len(gt_pairs))
    return float(np.mean(recalls)) if recalls else 0.0


def detection_metrics(predictions, ground_truth, iou_threshold=0.3):
    grouped = defaultdict(list)
    for image_id, box, label, score in predictions:
        grouped[image_id].append((box, label, score))
    all_tp = all_fp = all_fn = 0
    for image_id, gts in ground_truth.items():
        preds = grouped[image_id]
        tp, fp, fn, mis = match_predictions(
            [pred[0] for pred in preds], [pred[1] for pred in preds],
            [pred[2] for pred in preds], [gt[0] for gt in gts],
            [gt[1] for gt in gts], iou_threshold,
        )
        all_tp += len(tp)
        all_fp += len(fp) + len(mis)
        all_fn += len(fn) + len(mis)
    precision = all_tp / (all_tp + all_fp) if all_tp + all_fp else 0.0
    recall = all_tp / (all_tp + all_fn) if all_tp + all_fn else 0.0
    thresholds = np.arange(0.5, 1.0, 0.05)
    return {
        "mAP@50": round(float(compute_map(predictions, ground_truth, 0.5)), 4),
        "mAP@50:95": round(float(np.mean([
            compute_map(predictions, ground_truth, threshold) for threshold in thresholds
        ])), 4),
        "mAR@50": round(compute_mar(predictions, ground_truth, 0.5), 4),
        "mAR@50:95": round(float(np.mean([
            compute_mar(predictions, ground_truth, threshold) for threshold in thresholds
        ])), 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(2 * precision * recall / (precision + recall), 4)
        if precision + recall else 0.0,
        "Total TP": all_tp,
        "Total FP": all_fp,
        "Total FN": all_fn,
    }


def binary_metrics(counts):
    tp, fp, fn, tn = (counts[key] for key in ("TP", "FP", "FN", "TN"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    total = tp + fp + fn + tn
    return {
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(2 * precision * recall / (precision + recall), 4)
        if precision + recall else 0.0,
        "Accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "Specificity": round(tn / (tn + fp), 4) if tn + fp else 0.0,
        **{f"Total {key}": counts[key] for key in ("TP", "FP", "FN", "TN")},
    }