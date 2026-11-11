import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.preprocess"

from ..disk.params import get_data_config, get_global_config, get_preprocess_config
from ..disk.seed import set_seed
from .class_filter import filter_classes, validate_class_selection
from .fusion import _fuse_splitted_datasets


def _process_single_source(input_dir: Path, output_dir: Path, cfg, full: bool = False) -> dict:
    """Función auxiliar ejecutada en un proceso independiente para cada fuente."""
    from .incremental import process_source

    stats = process_source(input_dir, output_dir, cfg, get_global_config(), full=full)
    print(f"Preprocesado completado: {output_dir}")
    return stats


def _format_stats(stats: dict) -> str:
    labels = (
        ("new", "nuevas"), ("modified", "modificadas"), ("deleted", "eliminadas"),
        ("moved", "cambiadas de split"), ("reused", "sin cambios"), ("failed", "fallidas"),
        ("tiles_written", "tiles escritos"), ("tiles_deleted", "tiles borrados"),
        ("augmentations_written", "aumentos escritos"),
    )
    return ", ".join(f"{label}={stats.get(key, 0)}" for key, label in labels)


def run_preprocess(full: bool = False):
    data_cfg = get_data_config()
    cfg = get_preprocess_config()
    validate_class_selection(cfg.classes)
    if not data_cfg.sources:
        raise ValueError("At least one data source is required for preprocessing.")
    set_seed(get_global_config().seed)

    input_dirs = data_cfg.sources
    output_dirs = []
    for idx, input_dir in enumerate(input_dirs):
        output_dirs.append(data_cfg.training_dataset_path / "sources" / f"{input_dir.name}_source{idx}")

    from .splitter import validate_ratios
    validate_ratios(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)

    print(f"Iniciando preprocesado en paralelo para {len(input_dirs)} fuentes...")

    # Ejecutar en paralelo usando ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        # Lanzamos una tarea por cada par de input/output
        futures = {
            executor.submit(_process_single_source, inp, out, cfg, full): out
            for inp, out in zip(input_dirs, output_dirs, strict=True)
        }

        # Esperamos a que terminen y controlamos posibles errores
        for future in as_completed(futures):
            out_path = futures[future]
            try:
                stats = future.result()  # Si hubo una excepción dentro del proceso, se relanzará aquí
            except Exception as e:
                traceback.print_exc()
                print(f"Error procesando {out_path}: {e}", file=sys.stderr)
                return 1
            if isinstance(stats, dict):
                print(f"{out_path.name}: {_format_stats(stats)}")

    _fuse_splitted_datasets(output_dirs, data_cfg.training_dataset_path)
    filter_classes(data_cfg.training_dataset_path, cfg.classes)
    print("Preprocessing, dataset fusion and class selection completed.")
    return 0
