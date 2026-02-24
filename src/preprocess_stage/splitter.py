"""
splitter.py
===========
Clase `Splitter` para dividir datasets COCO de segmentación de instancias
(polígonos) en splits de entrenamiento, validación y test manteniendo
la distribución de categorías original mediante estratificación multi-etiqueta.

Estructura esperada del dataset:
    dataset/
    ├── images/
    │   ├── img1.jpg
    │   └── ...
    └── annotations.json   ← archivo COCO

Uso:
    splitter = Splitter(
        dataset_dir="path/to/dataset",
        output_dir="path/to/output",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        copy_images=True,
    )
    splits = splitter.split()
"""

from __future__ import annotations

import json
import yaml
import logging
from logger import get_logger
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import numpy as np

logger = get_logger(__name__, level=logging.DEBUG)

# ---------------------------------------------------------------------------
# Utilidades de estratificación multi-etiqueta
# ---------------------------------------------------------------------------

def _multilabel_stratified_split(
    image_ids: List[int],
    label_matrix: np.ndarray,
    ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
    Divide `image_ids` en dos subsets de tamaño aproximado `ratio` y
    `1-ratio` usando estratificación iterativa multi-etiqueta
    (Sechidis et al., 2011).

    Si `iterstrat` no está instalado, recurre a una versión simplificada
    basada en muestreo estratificado por categoría más frecuente.
    """

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=1 - ratio, random_state=seed
    )
    idx_a, idx_b = next(
        splitter.split(np.zeros(len(image_ids)), label_matrix)
    )
    ids_array = np.array(image_ids)
    return ids_array[idx_a].tolist(), ids_array[idx_b].tolist()


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class Splitter:
    """
    Divide un dataset COCO de segmentación de instancias en splits de
    entrenamiento, validación y test manteniendo la distribución de
    categorías mediante estratificación multi-etiqueta.

    Parameters
    ----------
    dataset_dir : str | Path
        Directorio raíz del dataset.
    output_dir : str | Path
        Directorio donde se guardarán los splits resultantes.
    train_ratio : float
        Proporción de datos para entrenamiento (default 0.7).
    val_ratio : float
        Proporción de datos para validación (default 0.15).
    test_ratio : float
        Proporción de datos para test (default 0.15).
    seed : int
        Semilla para reproducibilidad (default 42).
    copy_images : bool
        Si True, copia físicamente las imágenes a las subcarpetas de output.
        Si False, solo genera los JSONs COCO (más rápido, útil en pruebas).
    """

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path = "output",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        image_action: str = "copy",  # "copy", "move" o "ignore"
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.annotations_path = self.dataset_dir / "_annotations.coco.json"
        self.images_dir = self.dataset_dir
        self.output_dir = Path(output_dir)
        self.seed = seed

        if image_action not in ("copy", "move", "ignore"):
            raise ValueError("image_action debe ser 'copy', 'move' o 'ignore'")
        self.image_action = image_action

        self._validate_ratios(train_ratio, val_ratio, test_ratio)
        self.ratios = {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        }

        self._coco: Optional[Dict] = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Path]:
        """
        Ejecuta la división del dataset.

        Returns
        -------
        Dict[str, Path]
            Diccionario {split_name: path_to_coco_json} con los paths de
            los archivos de anotaciones generados.
        """
        logger.info("Cargando anotaciones COCO...")
        coco = self._load_coco()

        logger.info("Construyendo matriz de etiquetas...")
        image_ids, label_matrix, category_count = self._build_label_matrix(coco)

        logger.info(
            f"Dataset: {len(image_ids)} imágenes | "
            f"{category_count} categorías | "
            f"{len(coco['annotations'])} anotaciones"
        )

        logger.info("Estratificando splits...")
        split_ids = self._stratify(image_ids, label_matrix)

        self._log_distribution(coco, split_ids, label_matrix, image_ids)

        logger.info("Generando archivos de salida...")
        output_paths = self._write_splits(coco, split_ids)

        logger.info("División completada.")
        for name, path in output_paths.items():
            logger.info(f"  [{name:>5}] → {path}")

        return output_paths

    # ------------------------------------------------------------------
    # Validación
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_ratios(train: float, val: float, test: float) -> None:
        total = train + val + test
        if not (0.999 < total < 1.001):
            raise ValueError(
                f"La suma de train+val+test debe ser 1.0, "
                f"pero se obtuvo {total:.4f}."
            )
        for name, ratio in [("train", train), ("val", val), ("test", test)]:
            if not 0 < ratio < 1:
                raise ValueError(
                    f"El ratio `{name}` debe estar en (0, 1), "
                    f"pero se obtuvo {ratio}."
                )

    # ------------------------------------------------------------------
    # Carga de datos
    # ------------------------------------------------------------------

    def _load_coco(self) -> Dict:
        if not self.annotations_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de anotaciones: {self.annotations_path}"
            )
        with open(self.annotations_path, encoding="utf-8") as f:
            coco = json.load(f)

        required_keys = {"images", "annotations", "categories"}
        missing = required_keys - set(coco.keys())
        if missing:
            raise ValueError(
                f"El archivo COCO no contiene las claves requeridas: {missing}"
            )

        return coco

    # ------------------------------------------------------------------
    # Construcción de la matriz de etiquetas
    # ------------------------------------------------------------------

    def _build_label_matrix(
        self, coco: Dict
    ) -> Tuple[List[int], np.ndarray, int]:
        """
        Construye una matriz binaria (n_images × n_categories) donde
        cada celda indica si la imagen contiene al menos una anotación
        de esa categoría.
        """
        categories = coco["categories"]
        cat_ids = [c["id"] for c in categories]
        cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
        n_cats = len(cat_ids)

        # Mapa imagen_id → índice en la lista
        image_ids = [img["id"] for img in coco["images"]]
        img_id_to_idx = {img_id: idx for idx, img_id in enumerate(image_ids)}

        label_matrix = np.zeros((len(image_ids), n_cats), dtype=np.int8)

        for ann in coco["annotations"]:
            img_idx = img_id_to_idx.get(ann["image_id"])
            cat_idx = cat_id_to_idx.get(ann["category_id"])
            if img_idx is not None and cat_idx is not None:
                label_matrix[img_idx, cat_idx] = 1

        # Imágenes sin ninguna anotación reciben un vector de ceros;
        # las tratamos como su propia "clase" para que no queden fuera.
        return image_ids, label_matrix, n_cats

    # ------------------------------------------------------------------
    # Estratificación
    # ------------------------------------------------------------------

    def _stratify(
        self, image_ids: List[int], label_matrix: np.ndarray
    ) -> Dict[str, List[int]]:
        """
        Divide los image_ids en train/val/test manteniendo la distribución.
        Estrategia de dos pasos:
          1. Separar (train+val) vs test
          2. Separar train vs val dentro del primer grupo
        """
        train_val_ratio = self.ratios["train"] + self.ratios["val"]
        test_ratio = self.ratios["test"]

        # Paso 1: (train+val) | test
        ids_train_val, ids_test = _multilabel_stratified_split(
            image_ids, label_matrix, train_val_ratio, seed=self.seed
        )

        # Submatriz de etiquetas para el grupo train+val
        idx_map = {img_id: i for i, img_id in enumerate(image_ids)}
        tv_indices = [idx_map[img_id] for img_id in ids_train_val]
        label_matrix_tv = label_matrix[tv_indices]

        # Proporción relativa de train dentro de train+val
        relative_train_ratio = self.ratios["train"] / train_val_ratio

        # Paso 2: train | val
        ids_train, ids_val = _multilabel_stratified_split(
            ids_train_val,
            label_matrix_tv,
            relative_train_ratio,
            seed=self.seed + 1,
        )

        return {"train": ids_train, "val": ids_val, "test": ids_test}

    # ------------------------------------------------------------------
    # Logging de distribución
    # ------------------------------------------------------------------

    def _log_distribution(
        self,
        coco: Dict,
        split_ids: Dict[str, List[int]],
        label_matrix: np.ndarray,
        image_ids: List[int],
    ) -> None:
        cat_names = {c["id"]: c["name"] for c in coco["categories"]}
        cat_ids = [c["id"] for c in coco["categories"]]
        idx_map = {img_id: i for i, img_id in enumerate(image_ids)}

        logger.info("\n--- Distribución por split ---")
        header = f"{'Categoría':<25}" + "".join(
            f"{'Total':>8}"
            + "".join(f"{s:>8}" for s in self.SPLITS)
        )
        logger.info(header)

        for cat_idx, cat_id in enumerate(cat_ids):
            total = int(label_matrix[:, cat_idx].sum())
            row = f"{cat_names[cat_id]:<25}"
            row += f"{total:>8}"
            for split in self.SPLITS:
                ids = split_ids[split]
                indices = [idx_map[i] for i in ids if i in idx_map]
                count = int(label_matrix[indices, cat_idx].sum())
                pct = count / total * 100 if total > 0 else 0
                row += f"  {count}({pct:.0f}%)"
            logger.info(row)

        logger.info(f"\n{'Imágenes totales':<25}{len(image_ids):>8}")
        for split in self.SPLITS:
            logger.info(f"  {split}: {len(split_ids[split])} imágenes")

    # ------------------------------------------------------------------
    # Escritura de los splits
    # ------------------------------------------------------------------

    def _write_splits(
        self, coco: Dict, split_ids: Dict[str, List[int]]
    ) -> Dict[str, Path]:
        """
        Genera para cada split:
          output_dir/
          ├── train/
          │   ├── images/     (si copy_images=True)
          │   └── annotations.json
          ├── val/
          │   └── ...
          └── test/
              └── ...
        """
        # Índices auxiliares
        img_id_to_info = {img["id"]: img for img in coco["images"]}

        # Anotaciones agrupadas por imagen
        anns_by_image: Dict[int, List[Dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        output_paths: Dict[str, Path] = {}

        for split_name, ids in split_ids.items():
            split_dir = self.output_dir / split_name
            images_out_dir = split_dir
            split_dir.mkdir(parents=True, exist_ok=True)

            split_images = [img_id_to_info[i] for i in ids if i in img_id_to_info]
            split_annotations = []
            for img_id in ids:
                split_annotations.extend(anns_by_image.get(img_id, []))

            split_coco = {
                "info": coco.get("info", {}),
                "licenses": coco.get("licenses", []),
                "categories": coco["categories"],
                "images": split_images,
                "annotations": split_annotations,
            }

            ann_path = split_dir / "_annotations.coco.json"
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(split_coco, f, ensure_ascii=False, indent=2)

            if self.image_action in ("copy", "move"):
                images_out_dir.mkdir(parents=True, exist_ok=True)
                self._process_images(split_images, images_out_dir)

            output_paths[split_name] = ann_path

        return output_paths

    def _process_images(
        self, images: List[Dict], dest_dir: Path
    ) -> None:
        """Copia o mueve las imágenes al directorio de destino."""
        not_found = 0
        for img_info in images:
            file_name = img_info.get("file_name", "")
            src = self.images_dir / file_name

            # Fallback: buscar solo por el nombre de archivo sin subcarpeta
            if not src.exists():
                src = self.images_dir / Path(file_name).name

            if src.exists():
                if self.image_action == "move":
                    shutil.move(src, dest_dir / src.name)
                else:
                    shutil.copy2(src, dest_dir / src.name)
            else:
                not_found += 1
                logger.debug(f"Imagen no encontrada: {src}")

        if not_found:
            logger.warning(
                f"{not_found} imágenes no encontradas en {self.images_dir}."
            )


# ---------------------------------------------------------------------------
# Entrypoint de línea de comandos
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Divide un dataset COCO en splits train/val/test "
        "con estratificación multi-etiqueta."
    )
    parser.add_argument("dataset_dir", help="Directorio raíz del dataset")
    parser.add_argument(
        "--annotations-file",
        default="annotations.json",
        help="Nombre del archivo de anotaciones COCO (default: annotations.json)",
    )
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Subcarpeta de imágenes (default: images)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directorio de salida (default: output)",
    )
    parser.add_argument(
        "--train", type=float, default=0.7, help="Ratio de entrenamiento (default: 0.7)"
    )
    parser.add_argument(
        "--val", type=float, default=0.15, help="Ratio de validación (default: 0.15)"
    )
    parser.add_argument(
        "--test", type=float, default=0.15, help="Ratio de test (default: 0.15)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Semilla aleatoria (default: 42)"
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="No copiar imágenes; solo generar los JSONs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    with open("params.yaml", mode="r") as f:
        params = yaml.safe_load(f)

    ratios = (0.7, 0.15, 0.15)
    if sum(ratios) != 1: raise Exception("Ratios must sum up to 1.0!")

    splitter = Splitter(
        dataset_dir=str(Path("data", params["taske-name"])),
        annotations_file=str(Path("data", params["taske-name"], "_annotations.coco.json")),
        images_dir=str(Path("data", params["taske-name"])),
        output_dir=str(Path("data", params["taske-name"])),
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        test_ratio=ratios[2],
        seed=params["seed"],
        copy_images=not args.no_copy,
    )
    splitter.split()