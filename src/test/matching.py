import numpy as np
from iou import compute_iou


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