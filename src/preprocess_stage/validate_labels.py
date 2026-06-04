"""
visualizar_coco_save.py

Script para visualizar imágenes y sus etiquetas (formato COCO) guardando las imágenes anotadas en disco,
ideal para entornos sin GUI (sin cv2.imshow).

Configura directamente las rutas y opciones dentro del script.
"""

import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np

# Configuración del usuario -----------------
ANNOTATIONS_PATH = r"D:\training_data\formatted\test\_annotations.coco.json"
IMAGES_DIR = r"D:\training_data\formatted\test"
SHOW_MASKS = True
RESIZE = 1.0
OUT_DIR = "label_validaton"  # directorio donde se guardan las imágenes anotadas
# -------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# Intentar cargar pycocotools para soporte de máscaras RLE
HAVE_PYCOCO = True
try:
    from pycocotools import mask as maskUtils
except Exception:
    HAVE_PYCOCO = False


def load_coco_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_index(coco_json, images_dir=None):
    images = {img['id']: img for img in coco_json.get('images', [])}
    categories = {cat['id']: cat for cat in coco_json.get('categories', [])}

    annotations_by_image = defaultdict(list)
    for ann in coco_json.get('annotations', []):
        annotations_by_image[ann['image_id']].append(ann)

    image_entries = []
    for img_id, img in images.items():
        fname = img.get('file_name')
        path = os.path.join(images_dir, fname) if images_dir else fname
        image_entries.append({
            'id': img_id,
            'file_name': fname,
            'path': path,
            'width': img.get('width'),
            'height': img.get('height')
        })

    image_entries.sort(key=lambda x: x['file_name'])
    return image_entries, categories, annotations_by_image


def random_color(seed=None):
    rnd = random.Random(seed)
    return tuple(int(rnd.random() * 255) for _ in range(3))


def draw_annotations(img_bgr, anns, categories, category_colors, show_masks=False, alpha=0.5):
    overlay = img_bgr.copy()
    out = img_bgr

    for ann in anns:
        cat_id = ann.get('category_id')
        cat = categories.get(cat_id, {'name': str(cat_id)})
        cat_name = cat.get('name', str(cat_id))
        color = category_colors.setdefault(cat_id, random_color(cat_id))

        bbox = ann.get('bbox')
        if bbox:
            x, y, w, h = bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{cat_name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        if show_masks and 'segmentation' in ann and ann['segmentation']:
            seg = ann['segmentation']
            if isinstance(seg, list):
                for poly in seg:
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    pts_int = np.round(pts).astype(np.int32)
                    cv2.fillPoly(overlay, [pts_int], color)
            elif isinstance(seg, dict) and HAVE_PYCOCO:
                mask = maskUtils.decode(seg)
                if mask.ndim == 3:
                    mask = mask[..., 0]
                colored_mask = np.zeros_like(img_bgr, dtype=np.uint8)
                colored_mask[mask == 1] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)

    if show_masks:
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    return out


def main():
    coco_json = load_coco_json(ANNOTATIONS_PATH)
    image_entries, categories, annotations_by_image = build_index(coco_json, images_dir=IMAGES_DIR)

    category_colors = {cid: random_color(cid) for cid in categories.keys()}

    saved_count = 0

    for idx, entry in enumerate(image_entries):
        path = entry['path']
        anns = annotations_by_image.get(entry['id'], [])

        # Skip images with no annotations
        if not anns:
            continue

        if not os.path.exists(path):
            print(f'Archivo no encontrado: {entry["file_name"]}')
            continue

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f'No se puede leer: {entry["file_name"]}')
            continue

        if RESIZE != 1.0:
            img = cv2.resize(img, (0,0), fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_AREA)

        annotated = draw_annotations(img, anns, categories, category_colors, show_masks=SHOW_MASKS)

        info = f"{saved_count+1}  {entry['file_name']}  anns:{len(anns)}"
        cv2.rectangle(annotated, (0,0), (800, 24), (0,0,0), -1)
        cv2.putText(annotated, info, (6,16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        out_path = os.path.join(OUT_DIR, f"vis_{saved_count:04d}.jpg")
        cv2.imwrite(out_path, annotated)
        print('Guardado:', out_path)

        saved_count += 1


if __name__ == '__main__':
    main()
