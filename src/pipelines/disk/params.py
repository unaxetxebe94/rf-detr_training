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
    images_per_worker: int = 4
    num_workers: int = 4


TILE_FORMATS = ("png", "jpeg", "raw")
CACHED_AUGMENTATION_BACKENDS = ("custom", "albumentations", "kornia")


@dataclass
class CachedConfig:
    cache_dir: Path | None = None
    tile_format: str = "png"
    jpeg_quality: int = 95
    negative_ratio: float = 1.0
    augmentation_backend: str = "custom"
    augment_prob: float = 0.5
    aug_config: dict | None = None
    num_workers: int = 4
    pin_memory: bool | None = None
    persistent_workers: bool | None = None
    prefetch_factor: int | None = None
    amp_dtype: str | None = "auto"
    eval_batch_size: int | None = None
    eval_interval: int = 1
    early_stopping: bool = False
    early_stopping_patience: int = 10
    max_bad_samples: int = 20

    def __post_init__(self):
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
        self.tile_format = str(self.tile_format).lower()
        if self.tile_format == "jpg":
            self.tile_format = "jpeg"
        self.augmentation_backend = str(self.augmentation_backend).lower()
        if self.tile_format not in TILE_FORMATS:
            raise ValueError(f"cached.tile-format debe ser uno de {TILE_FORMATS}.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("cached.jpeg-quality debe estar en [1, 100].")
        if not 0 <= self.negative_ratio <= 1:
            raise ValueError("cached.negative-ratio debe estar en [0, 1].")
        if self.augmentation_backend not in CACHED_AUGMENTATION_BACKENDS:
            raise ValueError(f"cached.augmentation-backend debe ser uno de {CACHED_AUGMENTATION_BACKENDS}.")
        if not 0 <= self.augment_prob <= 1:
            raise ValueError("cached.augment-prob debe estar en [0, 1].")
        if self.num_workers < 0 or self.eval_interval < 1 or self.max_bad_samples < 0:
            raise ValueError("cached.num-workers y max-bad-samples deben ser >= 0 y eval-interval >= 1.")

    def resolved_cache_dir(self, training_dataset_path: Path) -> Path:
        # Por defecto dentro del dataset: mismo volumen, así la vista puede usar hardlinks.
        return self.cache_dir or Path(training_dataset_path) / "_tile_cache"


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


def get_cached_config():
    params = read_params()
    values = {key.replace("-", "_"): value for key, value in (params.get("cached") or {}).items()}
    return CachedConfig(**values)

