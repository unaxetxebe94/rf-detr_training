import json
from PIL import Image
from pathlib import Path

import logging
from logger import get_logger

class LabelCorrector:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.logger = get_logger(__name__, level=logging.DEBUG)
        self.SPLITS = ["train", "valid", "test"]

    def run(self):
        for split in self.SPLITS:
            self._correct_split(split)

    def _correct_split(self, split: str) -> None:
        # Leemos las anotaciones
        with open(self.dataset_path / split / "_annotations.coco.json", mode="r") as f:
            coco = json.load(f)
        imgs = coco["images"]
        cats = coco["categories"]
        anns = coco["annotations"]

        # Obtenemos mapeos que vamos a usar
        new_cat_id_by_old = {cat["id"]: new_id for new_id, cat in enumerate(cats)}
        imgs_name_by_id = {img["id"]: img["file_name"] for img in imgs}

        # Reordenamos los ids de las categorías para que sean 0..num_classes - 1
        for cat in cats:
            cat["id"] = new_cat_id_by_old[cat["id"]]
        self.logger.info(f"El nuevo mapeo de categorías: {new_cat_id_by_old}")

        # Limitamos las bbox a las dimensiones de la imagen en este for
        new_anns = []
        for ann in anns:
            ann_img_name = imgs_name_by_id[ann["image_id"]]
            bbox = ann["bbox"]
            with Image.open(self.dataset_path / split / ann_img_name) as f:
                img_width = f.width
                img_height = f.height

            x, y, w, h = bbox

            changed = False
            if x < 0: x = 0; changed = True
            if x > img_width: x = img_width; changed = True

            if y < 0: y = 0; changed = True
            if y > img_height: y = img_height; changed = True

            if w < 0: w = 0; changed = True
            if x + w > img_width: w = img_width - x; changed = True

            if h < 0: h = 0; changed = True
            if y + h > img_height: h = img_height - y; changed = True

            if changed:
                self.logger.info(f"El bbox: {bbox} se ha cambiado a {[x, y, w, h]}")
                ann["bbox"] = [x, y, w, h]

            ann["category_id"] = new_cat_id_by_old[ann["category_id"]]  # Sí o sí reemplazamos la antigua versión de la categoría

            # Si la anotación es demasido pequeña, se elimina
            if w >= 4 or h >= 4: new_anns.append(ann)
            else: self.logger.info(f"Eliminado el bbox pequeño: {[x, y, w, h]}")

        # Añadimos información adicional al dataset
        coco["info"] = {
            "description": "Dataset used dor the training of the B1375 proyect in Biele Digital"
        }
        coco["licenses"] = ["Only usable by Biele Group or companies collaborating with Biele Group"]
        coco["images"] = imgs
        coco["categories"] = cats
        coco["annotations"] = new_anns

        # Guardamos el dataset correcto
        with open(self.dataset_path / split / "_annotations.coco.json", mode="w") as f:
            json.dump(coco, f)

        






            
            

