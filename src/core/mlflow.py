import json
import os
from contextlib import contextmanager
from pathlib import Path

# El proyecto usa el backend de ficheros (file:./mlflow); MLflow 3.18 lo exige explícito.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import datetime
import re
from collections.abc import Mapping
from dataclasses import dataclass

from mlflow.entities import Experiment
from pytorch_lightning.loggers import MLFlowLogger

import mlflow
from mlflow import ActiveRun

from .params import get_global_config, get_preprocess_config

_MLFLOW_DIR = Path(__file__).resolve().parents[2] / "mlflow"
MLFLOW_TRACKING_URI = _MLFLOW_DIR.as_uri()

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
    global _MLFLOW_CFG
    global_cfg = get_global_config()

    if mlflow.active_run() is not None:
        return
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(global_cfg.experiment_name)
    run = mlflow.start_run(run_name=get_run_name())

    _MLFLOW_CFG = MLFlowConfig(
        uri=MLFLOW_TRACKING_URI,
        experiment_name=global_cfg.experiment_name,
        experiment=experiment,
        run_id=run.info.run_id,
        run_name=run.info.run_name,
        active_run=run
    )
    print(
        f"MLflow: {MLFLOW_TRACKING_URI}; experimento={experiment.name}; "
        f"run_id={run.info.run_id}"
    )


@contextmanager
def mlflow_run(stage: str):
    """Close only runs created here; nested pipeline stages reuse their owner."""
    owns_run = mlflow.active_run() is None
    if owns_run:
        initialize_mlflow()
        mlflow.set_tag("stage", stage)
    mlflow.set_tag(f"stage_{stage}", True)
    try:
        yield mlflow.active_run()
    except BaseException:
        if owns_run:
            stop_mlflow(status="FAILED")
        raise
    else:
        if owns_run:
            stop_mlflow()


def get_mlflow_cfg() -> MLFlowConfig | None:
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
    if mlflow.active_run() is None:
        initialize_mlflow()
    run = mlflow.active_run()
    return MLFlowLogger(
        experiment_name=get_global_config().experiment_name,
        run_name=run.info.run_name,
        run_id=run.info.run_id,
        tracking_uri=MLFLOW_TRACKING_URI,
        save_dir=str(_MLFLOW_DIR),
    )


def _sanitize_key(name: str) -> str:
    """Convierte un nombre en una clave válida para métricas/params de MLflow."""
    name = name.replace("@", "_at_").replace(":", "-")
    return _INVALID_KEY_CHARS.sub("_", name)


def _ensure_active_run(stage: str | None = None) -> None:
    if mlflow.active_run() is None:
        initialize_mlflow()
    if stage is not None:
        mlflow.set_tag(f"stage_{stage}", True)


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
    global _MLFLOW_CFG
    active_run = mlflow.active_run()

    if active_run is not None:
        mlflow.end_run(status=status)

    _MLFLOW_CFG = None
