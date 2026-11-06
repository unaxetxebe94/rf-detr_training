import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.preprocess"

from ..core.params import get_data_config, get_global_config, get_preprocess_config
from .annotations import read_annotations


def _process_single_source(input_dir: Path, output_dir: Path, cfg) -> None:
    """Función auxiliar ejecutada en un proceso independiente para cada fuente."""
    from .annotations import correct_labels
    from .augmenter import augment_dataset
    from .splitter import split_annotations
    from .tile_creation import create_tiles

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_annotations(read_annotations(input_dir / "_annotations.coco.json"))

    for split, annotations in splits.items():
        create_tiles(
            output_dir / split,
            input_dir=input_dir,
            annotations=annotations,
            saving_prob=cfg.saving_prob if split == "train" else 1.0,
            save_source_images=split == "test",
        )

    augment_dataset(output_dir / "train")
    correct_labels(output_dir)
    print(f"Preprocesado completado: {output_dir}")


def main(argv: list[str] | None = None) -> int:
    data_cfg = get_data_config()
    cfg = get_preprocess_config()
    random.seed(get_global_config().seed)

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
            executor.submit(_process_single_source, inp, out, cfg): out
            for inp, out in zip(input_dirs, output_dirs, strict=True)
        }

        # Esperamos a que terminen y controlamos posibles errores
        for future in as_completed(futures):
            out_path = futures[future]
            try:
                future.result()  # Si hubo una excepción dentro del proceso, se relanzará aquí
            except Exception as e:
                print(f"Error procesando {out_path}: {e}", file=sys.stderr)
                return 1

    print("Todos los preprocesados en paralelo han finalizado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
