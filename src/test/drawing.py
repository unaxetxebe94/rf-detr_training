import cv2
import numpy as np

from .iou import box_xywh_to_xyxy


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