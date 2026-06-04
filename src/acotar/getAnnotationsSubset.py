import json
import argparse
from pathlib import Path
from collections import defaultdict

def filter_coco_by_classes(root_dir: Path, class_names: list):

    input_path = root_dir / "_annotations.coco.json"
    cp = root_dir / "_annotations_allclasses.coco.json"

    with open(input_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # Mapear nombre de clase -> id original
    name_to_id = {cat["name"]: cat["id"] for cat in coco["categories"]}

    # Validar clases
    missing = [c for c in class_names if c not in name_to_id]
    if missing:
        raise ValueError(f"Clases no encontradas: {missing}")

    selected_ids = {name_to_id[c] for c in class_names}

    # Filtrar categorías
    new_categories = [cat for cat in coco["categories"] if cat["id"] in selected_ids]

    # Reindexar categorías (opcional pero limpio)
    old_to_new_id = {cat["id"]: i for i, cat in enumerate(new_categories)}
    for cat in new_categories:
        cat["id"] = old_to_new_id[cat["id"]]

    # Filtrar anotaciones
    new_annotations = []
    used_image_ids = set()

    for ann in coco["annotations"]:
        if ann["category_id"] in selected_ids:
            ann_copy = ann.copy()
            ann_copy["category_id"] = old_to_new_id[ann["category_id"]]
            new_annotations.append(ann_copy)
            used_image_ids.add(ann["image_id"])

    # Filtrar imágenes (solo las que tienen anotaciones)
    # new_images = [img for img in coco["images"] if img["id"] in used_image_ids]

    # Construir nuevo COCO
    new_coco = {
        "images": coco["images"],  # Mantener todas las imágenes, incluso las sin anotaciones
        "annotations": new_annotations,
        "categories": new_categories
    }

    # Copiar campos extra si existen
    for key in coco:
        if key not in new_coco:
            new_coco[key] = coco[key]

    # Guardar dataset viejo
    with open(cp, 'w', encoding='utf-8') as f:
        json.dump(coco, f, indent=4, ensure_ascii=False)

    # Guardar dataset filtrado
    with open(input_path, 'w', encoding='utf-8') as f:
        json.dump(new_coco, f, indent=4, ensure_ascii=False)

    print(f"Nuevo dataset guardado en: {cp}")
    print(f"Imágenes: {len(coco["images"])}")
    print(f"Anotaciones: {len(new_annotations)}")
    print(f"Clases: {[c['name'] for c in new_categories]}")


if __name__ == "__main__":
    root_dir = Path(r"E:\rf-detr_training\data\formatted")
    classes = ["Damaged face", "Broken lipping", "Foil Tear", "Short lipping"]

    for split in ["train", "valid", "test"]:
        filter_coco_by_classes(root_dir / split, classes)