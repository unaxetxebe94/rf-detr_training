import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class GlobalConfig:
    task_name: str
    model_type: str
    seed: int
    data_src1: Path
    data_src2: Path
    final_data: Path


@dataclass
class PreprocessConfig:
    requires_preprocess: bool
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


@dataclass
class TrainConfig:
    lr: float
    lr_encoder: float
    batch_size: int
    grad_accum_steps: int
    epochs: int
    weight_decay: float
    tensorboard: bool
    wandb: bool
    project: str
    mlflow: bool
    clearml: bool
    run_test: bool
    eval_max_dets: int


def read_params(path: str = r"params.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_global_config():
    params = read_params()
    values = params.get("global", params)
    values = {key.replace("-", "_"): value for key, value in values.items()}
    return GlobalConfig(
        task_name=values["task_name"],
        model_type=values["model_type"].lower(),
        seed=values["seed"],
        data_src1=Path(values["data_src1"]),
        data_src2=Path(values["data_src2"]),
        final_data=Path(values["final_data"]),
    )


def get_preprocess_config():
    params = read_params()
    values = {key.replace("-", "_"): value for key, value in params["preprocess"].items()}
    values.setdefault("min_retained_dimension_ratio", 0.10)
    return PreprocessConfig(**values)

def get_train_config():
    params = read_params()
    return TrainConfig(**params["train"])