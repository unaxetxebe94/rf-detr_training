import json
import os
import json
import shutil
import logging 
from logger import get_logger
from utils import read_params, save_mapping
from pathlib import Path

class DatasetFuser:

    @staticmethod
    def validate_dataset(dataset_path):
        # El dataset existe
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at path: {dataset_path}")
        # El path es un directorio
        if not os.path.isdir(dataset_path):
            raise ValueError(f"Provided dataset path is not a directory: {dataset_path}")
        # El dataset contiene anotaciones
        annotations_path = os.path.join(dataset_path, "_annotations.coco.json")
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations file not found at path: {annotations_path}")


    @staticmethod
    def copy_dataset_images(dataset_path, output_path):
        img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        
        files = os.listdir(dataset_path)
        
        for file in files:
            if file.lower().endswith(img_extensions):
                src = os.path.join(dataset_path, file)
                dst = os.path.join(output_path, file)
                shutil.copy2(src, dst)


    @staticmethod
    def fuse_datasets(split1_path, split2_path, output_path):
        # Validar datasets
        DatasetFuser.validate_dataset(split1_path)
        DatasetFuser.validate_dataset(split2_path)

        # Cargar anotaciones
        with open(os.path.join(split1_path, "_annotations.coco.json"), 'r') as f:
            annotations1 = json.load(f)
        with open(os.path.join(split2_path, "_annotations.coco.json"), 'r') as f:
            annotations2 = json.load(f)

        # Fusionar categorías
        category_mapping = {}
        fused_categories = annotations1.get('categories', [])[:]
        
        # Manejar caso de lista vacía con default=0
        current_max_cat_id = max((cat['id'] for cat in fused_categories), default=0)
        next_category_id = current_max_cat_id + 1

        fused_category_names = {c['name']: c for c in fused_categories}

        for cat in annotations2.get('categories', []):
            if cat['name'] not in fused_category_names:
                new_cat = cat.copy()
                new_cat['id'] = next_category_id
                category_mapping[cat['id']] = next_category_id
                fused_categories.append(new_cat)
                fused_category_names[cat['name']] = new_cat # Actualizar cache local
                next_category_id += 1
            else:
                existing_cat = fused_category_names[cat['name']]
                category_mapping[cat['id']] = existing_cat['id']

        # Fusionar imágenes y anotaciones
        fused_images = annotations1.get('images', [])[:]
        fused_annotations = annotations1.get('annotations', [])[:]
        
        # Manejar caso de listas vacías con default=0
        next_image_id = max((img['id'] for img in fused_images), default=0) + 1
        next_annotation_id = max((ann['id'] for ann in fused_annotations), default=0) + 1

        for img in annotations2.get('images', []):
            new_img = img.copy()
            # Guardamos el ID antiguo para mapear las anotaciones
            old_img_id = img['id']
            
            new_img['id'] = next_image_id
            fused_images.append(new_img)

            # Filtrar anotaciones correspondientes a esta imagen
            img_annotations = [ann for ann in annotations2.get('annotations', []) if ann['image_id'] == old_img_id]
            
            for ann in img_annotations:
                new_ann = ann.copy()
                new_ann['id'] = next_annotation_id
                new_ann['image_id'] = next_image_id # Usar el nuevo ID de la imagen
                
                # Verificar si la categoría existe en el mapeo (por seguridad)
                if ann['category_id'] in category_mapping:
                    new_ann['category_id'] = category_mapping[ann['category_id']]
                    fused_annotations.append(new_ann)
                    next_annotation_id += 1
                
            next_image_id += 1

        # Crear el dataset fusionado
        fused_dataset = {
            'info': {
                "creator": "Biele Digital"
            },
            'licenses': ['Only usable for members of Biele Group.'],
            'categories': fused_categories,
            'images': fused_images,
            'annotations': fused_annotations
        }
        
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "_annotations.coco.json"), 'w') as f:
            json.dump(fused_dataset, f, indent=4) # Indent para legibilidad

        # Copiar imágenes
        DatasetFuser.copy_dataset_images(split1_path, output_path)
        DatasetFuser.copy_dataset_images(split2_path, output_path)

    
    @staticmethod
    def fuse_splitted_datasets(dataset1_path, dataset2_path, output_path):
        splits = ["test", "train", "valid"]
        for split in splits:
            split1_path = os.path.join(dataset1_path, split)
            split2_path = os.path.join(dataset2_path, split)
            split_output_path = os.path.join(output_path, split)
            DatasetFuser.fuse_datasets(split1_path, split2_path, split_output_path)



def _count_split_stats(dataset_dir: Path, splits: tuple = ("train", "val", "test")) -> dict:
    """
    Cuenta imágenes y anotaciones en cada split del dataset formateado.
    Devuelve un dict {split: {n_images, n_annotations}}.
    """
    stats = {}
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue

        # Contar imágenes
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".tiff"}
        n_images = sum(
            1 for f in split_dir.iterdir()
            if f.suffix.lower() in image_exts
        )

        # Contar anotaciones desde el COCO JSON si existe
        annotations_file = split_dir / "_annotations.coco.json"
        n_annotations = 0
        if annotations_file.exists():
            with open(annotations_file) as f:
                coco = json.load(f)
            n_annotations = len(coco.get("annotations", []))

        stats[split] = {"n_images": n_images, "n_annotations": n_annotations}
        logger.debug(f"Split '{split}': {n_images} imágenes, {n_annotations} anotaciones")

    return stats

def save_preprocess_info(params: dict, dataset_dir: Path, output_dir: Path) -> None:
    """
    Guarda un JSON con los parámetros de preprocesado y las estadísticas del
    dataset resultante. Este fichero lo lee process_results.py para logearlo en W&B.
    """
    preprocess_params = params.get("preprocess", {})

    # Estadísticas de los splits generados
    split_stats = _count_split_stats(dataset_dir)

    info = {
        "params": {
            "resize":                    preprocess_params.get("resize"),
            "apply_roi":                 preprocess_params.get("apply-roi"),
            "saving_prob":               preprocess_params.get("saving-prob"),
            "train_ratio":               preprocess_params.get("train-ratio"),
            "val_ratio":                 preprocess_params.get("val-ratio"),
            "test_ratio":                preprocess_params.get("test-ratio"),
            "augmentations_per_image":   preprocess_params.get("augmentations-per-image"),
            "max_transforms_per_sample": preprocess_params.get("max-transforms-per-sample"),
            "model_type":                params.get("model-type"),
            "seed":                      params.get("seed"),
            "data_src":                  params.get("data-src"),
            "task_name":                 params.get("task-name"),
        },
        "split_stats": split_stats,
    }

    output_path = output_dir / "preprocess_info.json"
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"preprocess_info.json guardado → {output_path}")

if __name__ == "__main__":

    logger = get_logger(__name__, level=logging.DEBUG)

    params = read_params()

    if os.path.exists(Path("data", "preprocessed_src1")):
        d1_path = Path("data", "preprocessed_src1")
    else:
        d1_path =Path(params["data-src1"]) / "master"

    DatasetFuser.fuse_splitted_datasets(
        dataset1_path=d1_path,
        dataset2_path=Path(params["data-src2"]) / "slave",
        output_path=params["final-data"]
    )

    # Guardamos un mapping de cat_id --> cat_name para el test
    output_dir  = Path("trainings", "temp")

    save_mapping()

    # ── Guardar info de preprocesado para W&B ────────────────────────────────
    stats_dir = Path(params["final-data"])
    save_preprocess_info(params, stats_dir, output_dir)

    logger.debug("Se ha terminado la fusión de datasets")