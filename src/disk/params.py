from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    sources: list[Path]
    training_dataset_path: Path


@dataclass
class GlobalConfig:
    experiment_name: str
    model_type: str
    seed: int


@dataclass
class PreprocessConfig:
    saving_prob: float
    min_retained_dimension_ratio: float
    resize: float
    apply_roi: bool
    train_ratio: float
    test_ratio: float
    val_ratio: float
    augmentations_per_image: int
    max_transforms_per_sample: int
    n_jobs: int = 4
    classes: list[str] | None = None


@dataclass
class TrainConfig:
    lr: float
    lr_encoder: float
    batch_size: int
    grad_accum_steps: int
    epochs: int
    weight_decay: float
    tensorboard: bool


@dataclass
class StreamingConfig:
    images_per_worker: int = 2
    num_workers: int = 2


def read_params(path: str = r"./params.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_global_config():
    params = read_params()
    values = params.get("global", params)
    values = {key.replace("-", "_"): value for key, value in values.items()}
    return GlobalConfig(
        experiment_name=values["experiment_name"],
        model_type=values["model_type"].lower(),
        seed=values["seed"],
    )


def get_data_config():
    params = read_params()
    data_params = params.get("data", {})
    if not data_params:
        raise ValueError("No se encontraron parámetros de configuración de datos en params.yaml")
    if "sources" not in data_params or "training-dataset-path" not in data_params:
        raise ValueError("Los parámetros de configuración de datos deben incluir 'sources' y 'training-dataset-path'")
    sources = [Path(source) for source in data_params["sources"]]
    return DataConfig(
        sources=sources,
        training_dataset_path=Path(data_params["training-dataset-path"])
    )


def get_preprocess_config():
    params = read_params()
    values = {key.replace("-", "_"): value for key, value in params["preprocess"].items()}
    values.setdefault("min_retained_dimension_ratio", 0.10)
    return PreprocessConfig(**values)

def get_train_config():
    params = read_params()
    return TrainConfig(**params["train"])


def get_streaming_config():
    params = read_params()
    values = {key.replace("-", "_"): value for key, value in (params.get("streaming") or {}).items()}
    cfg = StreamingConfig(**values)
    if cfg.images_per_worker < 1 or cfg.num_workers < 0:
        raise ValueError("streaming.images-per-worker debe ser >= 1 y streaming.num-workers >= 0.")
    return cfg

