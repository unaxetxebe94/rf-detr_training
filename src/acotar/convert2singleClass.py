import json
import argparse
import copy

def convert_to_single_class(input_path, output_path, new_class_name="defecto"):
    # Cargar anotaciones COCO
    with open(input_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # Crear copia profunda para no modificar el original
    new_coco = copy.deepcopy(coco)

    # Definir nueva única categoría
    new_category = {
        "id": 0,
        "name": new_class_name,
        "supercategory": "none"
    }

    # Reemplazar categorías
    new_coco["categories"] = [new_category]

    # Modificar todas las anotaciones
    for ann in new_coco["annotations"]:
        ann["category_id"] = 0

    # Guardar nuevo JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_coco, f, indent=4, ensure_ascii=False)

    print(f"Archivo convertido guardado en: {output_path}")


if __name__ == "__main__":
    INPUT = r"E:\rf-detr_training\data\top_and_bottom\valid\_annotations.coco.json"
    OUTPUT = r"E:\rf-detr_training\data\top_and_bottom\valid\_annotations.coco.single_class.json"
    CLASS_NAME = "defect"

    convert_to_single_class(INPUT, OUTPUT, CLASS_NAME)