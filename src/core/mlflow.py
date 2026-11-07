import json
import os

# El proyecto usa el backend de ficheros (file:./mlflow); MLflow 3.18 lo exige explícito.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import re
import mlflow
from mlflow import ActiveRun
from mlflow.entities import Experiment
import datetime
from collections.abc import Mapping
from dataclasses import dataclass
from pytorch_lightning.loggers import MLFlowLogger

from .params import get_global_config, get_preprocess_config

_MLFLOW_DIR = "./mlflow"
MLFLOW_TRACKING_URI = f"file:{_MLFLOW_DIR}"

# MLflow solo acepta en métricas y params letras, números y "_/. - ": lo demás lo reemplazamos.
_INVALID_KEY_CHARS = re.compile(r"[^\w/. \-]", re.UNICODE)

@dataclass(frozen=True)
class MLFlowConfig:
    uri: str
    experiment_name: str
    experiment: Experiment
    run_id: str
    run_name: str
    active_run: ActiveRun

_MLFLOW_CFG: MLFlowConfig | None = None


def initialize_mlflow():
    global_cfg = get_global_config()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(global_cfg.experiment_name)
    run = mlflow.start_run()


    _MLFLOW_CFG = MLFlowConfig(
        uri=MLFLOW_TRACKING_URI,
        experiment_name=global_cfg.experiment_name,
        experiment=experiment,
        run_id=run.info.run_id,
        run_name=run.info.run_name,
        active_run=run
    )


def get_mlflow_cfg() -> MLFlowConfig:
    return _MLFLOW_CFG


def get_run_name() -> str:
    global_cfg = get_global_config()
    preprocess_cfg = get_preprocess_config()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        return (
            f"[{now}]-saving_prob={preprocess_cfg.saving_prob}-resize={preprocess_cfg.resize}"
            f"-apply_roi={preprocess_cfg.apply_roi}-train_ratio={preprocess_cfg.train_ratio}"
            f"-test_ratio={preprocess_cfg.test_ratio}-val_ratio={preprocess_cfg.val_ratio}"
            f"-augmentations_per_image={preprocess_cfg.augmentations_per_image}"
            f"-max_transformations_per_sample={preprocess_cfg.max_transforms_per_sample}"
            f"-model={global_cfg.model_type}"
        )
    except KeyError as ke:
        print(f"Error al leer el preprocesado: {ke}")
        print("Se asume que se está entrenando desde Collab")
        return f"[{now}]-Training"


def get_mlflow_logger() -> MLFlowLogger:
    return MLFlowLogger(
        experiment_name=_MLFLOW_CFG.experiment_name,
        run_name=_MLFLOW_CFG.run_name,
        run_id=_MLFLOW_CFG.run_id,
        save_dir=_MLFLOW_DIR
    )


def _sanitize_key(name: str) -> str:
    """Convierte un nombre en una clave válida para métricas/params de MLflow."""
    name = name.replace("@", "_at_").replace(":", "-")
    return _INVALID_KEY_CHARS.sub("_", name)


def _ensure_active_run(stage: str | None = None) -> None:
    if mlflow.active_run() is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(get_global_config().experiment_name)
        mlflow.start_run()
    if stage is not None:
        mlflow.set_tag("stage", stage)


def ensure_test_run() -> str:
    """Activa (o reutiliza) una run de MLflow para el test y devuelve su run id."""
    _ensure_active_run(stage="test")
    return mlflow.active_run().info.run_id


def log_dict_metrics(metrics: Mapping[str, float], prefix: str = "") -> None:
    """Loggea en la run activa un diccionario plano como métricas de MLflow."""
    if not metrics:
        return
    _ensure_active_run()
    mlflow.log_metrics({
        f"{prefix}{_sanitize_key(str(key))}": float(value)
        for key, value in metrics.items()
    })


def log_dict_params(params: Mapping[str, object], prefix: str = "") -> None:
    """Loggea en la run activa un diccionario como parámetros de MLflow (sin artefactos)."""
    if not params:
        return
    _ensure_active_run()
    mlflow.log_params({
        f"{prefix}{_sanitize_key(str(key))}": (
            value if isinstance(value, str) else json.dumps(value)
        )
        for key, value in params.items()
    })


def stop_mlflow(status: str = "FINISHED") -> None:
    active_run = mlflow.active_run()

    if active_run is not None:
        mlflow.end_run(status=status)

    _MLFLOW_CFG = None