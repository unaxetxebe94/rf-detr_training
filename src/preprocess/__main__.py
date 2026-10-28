import argparse
import os
import random
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.preprocess"

from ..params import get_global_config, get_preprocess_config
from .annotations import SPLITS


def is_dataset_formatted(dataset_dir: Path) -> bool:
    dataset_dir = Path(dataset_dir)
    return all(
        (dataset_dir / split).is_dir()
        and (dataset_dir / split / "_annotations.coco.json").is_file()
        for split in SPLITS
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocesado de datasets COCO.")
    parser.add_argument("--is-master", action="store_true")
    args = parser.parse_args(argv)
    global_cfg = get_global_config()
    cfg = get_preprocess_config()
    input_dir = global_cfg.data_src1 if args.is_master else global_cfg.data_src2
    output_dir = (
        global_cfg.final_data / "master"
        if args.is_master
        else global_cfg.data_src2 / "formatted"
    )

    if not cfg.requires_preprocess:
        if is_dataset_formatted(output_dir):
            print(f"Dataset ya formateado: {output_dir}")
        elif all(is_dataset_formatted(path) for path in (
            global_cfg.data_src1, global_cfg.data_src2
        )):
            print("Los datasets de entrada ya estan formateados.")
        else:
            raise FileNotFoundError(
                f"No se ha encontrado el dataset formateado en {output_dir} "
                f"ni ambos datasets en {global_cfg.data_src1} y {global_cfg.data_src2}."
            )
        return 0

    vips_bin = Path(r"C:\Program Files\vips-dev-8.17\bin")
    if os.name == "nt" and vips_bin.is_dir():
        os.environ["PATH"] = str(vips_bin) + os.pathsep + os.environ.get("PATH", "")

    from .augmenter import augment_dataset
    from .label_corrector import correct_labels
    from .splitter import split_dataset, validate_ratios
    from .tile_creation import create_tiles

    validate_ratios(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)
    random.seed(global_cfg.seed)
    print(f"Iniciando preprocesado: {input_dir} -> {output_dir}")
    create_tiles(output_dir, input_dir=input_dir)
    split_dataset(output_dir)
    augment_dataset(output_dir / "train")
    correct_labels(output_dir)
    print(f"Preprocesado completado: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
