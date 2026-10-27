import numpy as np
from collections import defaultdict

from iou import compute_iou


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