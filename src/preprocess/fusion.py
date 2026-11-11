import json
import logging
import os
import shutil
import sys
from pathlib import Path

from ..core.logger import get_logger
from ..core.params import get_data_config, get_preprocess_config
from .class_filter import filter_classes

logger = get_logger(__name__, level=logging.DEBUG)


def _normalize_category(category):
    normalized = {**category, "name": category["name"].lower()}
    if isinstance(normalized.get("supercategory"), str):
        normalized["supercategory"] = normalized["supercategory"].lower()
    return normalized


def _validate_dataset(dataset_path):
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


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


def _referenced_files(coco):
    """Rutas relativas (formato POSIX) de las imágenes que usa un JSON COCO."""
    files = [image["file_name"] for image in coco.get("images", [])]
    files.extend(source["file_name"] for source in coco.get("source_images", []) if "file_name" in source)
    return {Path(file_name).as_posix() for file_name in files}


def _copy_dataset_images(dataset_path, output_path, referenced):
    """Copia solo las imágenes referenciadas que falten o hayan cambiado (tamaño/mtime)."""
    dataset_path, output_path = Path(dataset_path), Path(output_path)
    copied = 0
    for file_name in sorted(referenced):
        src, dst = dataset_path / file_name, output_path / file_name
        if not src.is_file():
            continue
        src_stat = src.stat()
        try:
            dst_stat = dst.stat()
            if dst_stat.st_size == src_stat.st_size and dst_stat.st_mtime_ns == src_stat.st_mtime_ns:
                continue
        except FileNotFoundError:
            pass
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def _remove_unreferenced_images(output_path, referenced):
    """Borra imágenes del split fusionado (y de source_images) que el JSON ya no usa."""
    output_path = Path(output_path)
    removed = 0
    for directory in (output_path, output_path / "source_images"):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if (
                path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                and path.relative_to(output_path).as_posix() not in referenced
            ):
                path.unlink()
                removed += 1
    return removed


def _fuse_datasets(dataset_paths, output_path):
    """Fusiona uno o más datasets preprocesados y copia sus imágenes a output_path.

    El primer dataset conserva sus IDs salvo las categorías equivalentes por
    nombre, que se unifican. Cada dataset restante se fusiona secuencialmente:

    - Las categorías se casan por nombre en minúsculas; las nuevas se añaden con IDs
      secuenciales. Los IDs de categorías acumulados nunca cambian, y garantizan
      que el resultado es determinista: depende únicamente del orden de la lista.
    - Imágenes y anotaciones de los datasets posteriores reciben IDs nuevos;
      las anotaciones cuyas categorías no estén en el mapeo se descartan.
    - ``source_images`` (copias de puertas completas del split test) se acumula
      remapeando los category_id de sus anotaciones.
    """
    if not dataset_paths:
        raise ValueError("Se necesita al menos un dataset para fusionar")

    dataset_paths = [Path(path) for path in dataset_paths]

    # Validar datasets
    for dataset_path in dataset_paths:
        _validate_dataset(dataset_path)

    # Cargar anotaciones del dataset base (conserva sus IDs)
    with open(os.path.join(dataset_paths[0], "_annotations.coco.json"), encoding="utf-8") as f:
        base_annotations = json.load(f)

    fused_categories, fused_category_names, base_category_mapping = [], {}, {}
    for category in base_annotations.get("categories", []):
        category = _normalize_category(category)
        name = category["name"]
        if name not in fused_category_names:
            fused_categories.append(category)
            fused_category_names[name] = category
        base_category_mapping[category["id"]] = fused_category_names[name]["id"]
    fused_images = base_annotations.get('images', [])[:]
    referenced_by_dataset = [_referenced_files(base_annotations)]
    fused_annotations = [
        {**ann, "category_id": base_category_mapping[ann["category_id"]]}
        for ann in base_annotations.get("annotations", [])
        if ann["category_id"] in base_category_mapping
    ]
    fused_source_images = [
        {
            **source,
            "annotations": [
                {**ann, "category_id": base_category_mapping[ann["category_id"]]}
                for ann in source["annotations"] if ann["category_id"] in base_category_mapping
            ],
        }
        for source in base_annotations.get("source_images", [])
    ]

    # Manejar caso de listas vacías con default=0
    next_category_id = max((cat['id'] for cat in fused_categories), default=0) + 1
    next_image_id = max((img['id'] for img in fused_images), default=0) + 1
    next_annotation_id = max((ann['id'] for ann in fused_annotations), default=0) + 1

    logger.info(
        f"Dataset base: {dataset_paths[0]} "
        f"({len(fused_images)} imágenes, {len(fused_annotations)} anotaciones, "
        f"{len(fused_categories)} categorías)"
    )

    for dataset_path in dataset_paths[1:]:
        with open(os.path.join(dataset_path, "_annotations.coco.json"), encoding="utf-8") as f:
            annotations_next = json.load(f)
        referenced_by_dataset.append(_referenced_files(annotations_next))

        # Fusionar categorías
        category_mapping = {}
        for cat in annotations_next.get('categories', []):
            cat = _normalize_category(cat)
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
        for img in annotations_next.get('images', []):
            new_img = img.copy()
            # Guardamos el ID antiguo para mapear las anotaciones
            old_img_id = img['id']

            new_img['id'] = next_image_id
            fused_images.append(new_img)

            # Filtrar anotaciones correspondientes a esta imagen
            img_annotations = [ann for ann in annotations_next.get('annotations', []) if ann['image_id'] == old_img_id]

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

        src_images = annotations_next.get("source_images", [])
        for source in src_images:
            fused_source_images.append({
                **source,
                "annotations": [
                    {**ann, "category_id": category_mapping[ann["category_id"]]}
                    for ann in source["annotations"]
                ],
            })

        logger.info(
            f"Fusionado: {dataset_path} "
            f"({len(annotations_next.get('images', []))} imágenes, "
            f"{len(annotations_next.get('annotations', []))} anotaciones)"
        )

    # Crear el dataset fusionado
    fused_dataset = {
        'info': {
            "creator": "Biele Digital"
        },
        'licenses': ['Only usable for members of Biele Group.'],
        'categories': fused_categories,
        'images': fused_images,
        'annotations': fused_annotations,
        'source_images': fused_source_images,
    }

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "_annotations.coco.json"), 'w', encoding="utf-8") as f:
        json.dump(fused_dataset, f, indent=4) # Indent para legibilidad

    # Copiar solo imágenes nuevas o cambiadas y borrar las que ya no se usan
    copied = sum(
        _copy_dataset_images(dataset_path, output_path, referenced)
        for dataset_path, referenced in zip(dataset_paths, referenced_by_dataset, strict=True)
    )
    removed = _remove_unreferenced_images(output_path, set().union(*referenced_by_dataset))
    logger.info(f"{output_path}: {copied} imágenes copiadas, {removed} imágenes obsoletas borradas")


def _fuse_splitted_datasets(dataset_paths, output_path):
    """Fusiona N datasets divididos en splits (test/train/valid).

    Para cada split fusiona los N datasets
    (dataset_paths[i]/split) sobre output_path/split.
    """
    dataset_paths = [Path(path) for path in dataset_paths]
    splits = ["test", "train", "valid"]
    for split in splits:
        split_paths = [os.path.join(dataset_path, split) for dataset_path in dataset_paths]
        split_output_path = os.path.join(output_path, split)
        _fuse_datasets(split_paths, split_output_path)


def _source_sort_key(path: Path):
    """Ordena los datasets preprocesados por el índice del sufijo '*_source<i>'."""
    name, _, suffix = path.name.rpartition("_source")
    if suffix.isdigit():
        return (0, int(suffix), name)
    return (1, 0, path.name)


def _discover_preprocessed_datasets(training_dataset_path: Path) -> list[Path]:
    """Descubre los datasets ya preprocesados que se van a fusionar.

    El preprocesado (src/preprocess/__main__.py) escribe un dataset por cada
    fuente en ``<training-dataset-path>/sources/<nombre>_source<i>``; se
    devuelven solo los dirs ``*_source<i>`` con índice numérico, ordenados por
    el índice de fuente del sufijo.
    """
    sources_dir = Path(training_dataset_path) / "sources"
    if not sources_dir.is_dir():
        raise FileNotFoundError(
            f"No existe el directorio de datasets preprocesados: {sources_dir}"
        )

    dataset_paths = [
        path
        for path in sources_dir.iterdir()
        if path.is_dir() and path.name.rpartition("_source")[2].isdigit()
    ]
    if not dataset_paths:
        raise FileNotFoundError(
            f"No se ha encontrado ningún dataset preprocesado ('*_source*') en: {sources_dir}"
        )

    return sorted(dataset_paths, key=_source_sort_key)


def main():
    data_cfg = get_data_config()

    dataset_paths = _discover_preprocessed_datasets(data_cfg.training_dataset_path)
    logger.info(f"Se van a fusionar {len(dataset_paths)} datasets preprocesados:")
    for dataset_path in dataset_paths:
        logger.info(f"  - {dataset_path}")

    _fuse_splitted_datasets(
        dataset_paths=dataset_paths,
        output_path=data_cfg.training_dataset_path
    )
    filter_classes(data_cfg.training_dataset_path, get_preprocess_config().classes)

    logger.debug("Se ha terminado la fusión de datasets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
