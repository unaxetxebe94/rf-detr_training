import numpy as np
from .iou import compute_iou
from .matching import match_predictions


def _threshold_f1(predictions, ground_truth, thresholds, iou_threshold):
    tp = fp = fn = 0
    for image_id, gts in ground_truth.items():
        preds = [
            pred for pred in predictions.get(image_id, [])
            if pred[2] >= thresholds.get(pred[1], 0.5)
        ]
        matched, unmatched, missed, misclassified = match_predictions(
            [pred[0] for pred in preds], [pred[1] for pred in preds],
            [pred[2] for pred in preds], [gt[0] for gt in gts],
            [gt[1] for gt in gts], iou_threshold,
        )
        tp += len(matched)
        fp += len(unmatched) + len(misclassified)
        fn += len(missed) + len(misclassified)
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def select_thresholds(predictions, ground_truth, category_names,
                      iou_threshold=0.5, threshold_range=None):
    """Compare per-class and global thresholds using validation micro-F1."""
    if not any(ground_truth.values()):
        raise ValueError("Validation needs ground-truth annotations to calibrate thresholds.")
    candidates = [float(round(value, 4)) for value in (
        np.arange(0.05, 1.0, 0.05) if threshold_range is None else threshold_range
    )]
    if not candidates:
        raise ValueError("At least one confidence threshold is required.")
    per_class = find_optimal_thresholds(
        predictions, ground_truth, category_names, iou_threshold, candidates
    )
    per_class_f1 = _threshold_f1(predictions, ground_truth, per_class, iou_threshold)
    classes = (
        set(category_names)
        | {label for pairs in ground_truth.values() for _, label in pairs}
        | {label for preds in predictions.values() for _, label, _ in preds}
    )
    global_threshold, global_f1 = candidates[0], -1.0
    for threshold in candidates:
        f1 = _threshold_f1(
            predictions, ground_truth,
            dict.fromkeys(classes, threshold), iou_threshold,
        )
        if f1 > global_f1:
            global_threshold, global_f1 = threshold, f1
    global_threshold = float(round(global_threshold, 4))
    strategy = "global" if global_f1 >= per_class_f1 else "per_class"
    selected = (
        dict.fromkeys(classes, global_threshold) if strategy == "global"
        else {label: per_class.get(label, 0.5) for label in classes}
    )
    report = {
        "split": "valid",
        "metric": "micro-F1",
        "iou_threshold": iou_threshold,
        "per_class_thresholds": {
            category_names.get(label, str(label)): threshold
            for label, threshold in sorted(per_class.items())
        },
        "per_class_f1": per_class_f1,
        "global_threshold": global_threshold,
        "global_f1": global_f1,
        "selected_strategy": strategy,
    }
    print(
        f"\nValidation micro-F1: per-class={per_class_f1:.4f}, "
        f"global={global_f1:.4f} (threshold={global_threshold:.2f}); "
        f"selected={strategy}"
    )
    return selected, report


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
    sorted_preds_by_img = {
        img_id: sorted(preds, key=lambda pred: -pred[2])
        for img_id, preds in raw_preds_by_img.items()
    }

    best_thresholds = {}
    print("\nSearching optimal confidence threshold per class …")
    print(f"  Matching IoU >= {iou_threshold:.2f}")

    for cls in sorted(all_classes):
        best_f1 = -1.0
        best_thresh = 0.5
        best_counts = (0, 0, 0)

        for thresh in threshold_range:
            tp = fp = fn = 0

            for img_id, gt_pairs in gt_by_img.items():
                gt_boxes_cls = [b for b, l in gt_pairs if l == cls]
                preds_for_img = sorted_preds_by_img.get(img_id, [])
                pred_boxes_cls = [b for b, l, s in preds_for_img
                                  if l == cls and s >= thresh]

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
                best_counts = (tp, fp, fn)

        best_thresholds[cls] = float(round(best_thresh, 4))
        cls_name = category_id_to_name.get(cls, str(cls))
        tp, fp, fn = best_counts
        print(
            f"  {cls_name:<25} thresh={best_thresh:.2f}  F1={best_f1:.4f}"
            f"  class_id={cls}  TP={tp} FP={fp} FN={fn}"
        )
        if best_f1 == 0:
            print(
                "    WARNING: no true positives at any candidate threshold; "
                "the selected threshold is not a calibrated optimum. "
                "Check class IDs, boxes and checkpoint."
            )

    return best_thresholds