import numpy as np
from .iou import compute_iou


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