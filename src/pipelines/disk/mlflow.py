import json
import os
from contextlib import contextmanager
from pathlib import Path

# El fallback local usa el backend de ficheros (file:./mlflow); MLflow 3.18 lo exige explícito.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import datetime
import re
import subprocess
import urllib.request
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

from mlflow.entities import Experiment
from pytorch_lightning.loggers import MLFlowLogger

import mlflow
from mlflow import ActiveRun

from .params import get_data_config, get_global_config, get_preprocess_config, params_path

_REPO_DIR = Path(__file__).resolve().parents[3]
_MLFLOW_DIR = _REPO_DIR / "mlflow"
# Servidor central compartido por RF-DETR, clasificadores y Dinomaly (MLFLOW_TRACKING_URI lo sustituye).
DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"
MODEL_FAMILY = "rfdetr"
# Se resuelve en la primera run (ver resolve_tracking_uri); los tests pueden fijarlo directamente.
MLFLOW_TRACKING_URI: str | None = None
_TRACKING_FALLBACK = False

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


def _server_is_healthy(uri: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def resolve_tracking_uri() -> str:
    """MLFLOW_TRACKING_URI or the central server; falls back to ./mlflow if the server does not answer."""
    global MLFLOW_TRACKING_URI, _TRACKING_FALLBACK
    if MLFLOW_TRACKING_URI is not None:
        return MLFLOW_TRACKING_URI
    uri = os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    if uri.startswith(("http://", "https://")) and not _server_is_healthy(uri):
        fallback = _MLFLOW_DIR.as_uri()
        warnings.warn(
            f"El servidor de MLflow {uri} no responde; se usa el almacenamiento local {fallback}.",
            stacklevel=2,
        )
        uri, _TRACKING_FALLBACK = fallback, True
    MLFLOW_TRACKING_URI = uri
    return uri


def experiment_name(name: str | None = None) -> str:
    """Experiments are grouped per model family: ``rfdetr/<experiment-name>``."""
    name = name or get_global_config().experiment_name
    return name if name.startswith(f"{MODEL_FAMILY}/") else f"{MODEL_FAMILY}/{name}"


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=_REPO_DIR, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return None


def standard_tags(face_group: str) -> dict[str, str]:
    """Tags shared by every model family (rfdetr, classifier, dinomaly) on the central server."""
    try:
        dataset = str(get_data_config().training_dataset_path)
    except Exception:
        dataset = ""
    status = _git("status", "--porcelain")
    return {
        "model_family": MODEL_FAMILY,
        "face_group": face_group,
        "config_path": str(params_path().resolve()),
        "git_commit": _git("rev-parse", "HEAD") or "",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "",
        "git_dirty": str(bool(status)) if status is not None else "",
        "dataset": dataset,
        "tracking_fallback": str(_TRACKING_FALLBACK),
    }


def initialize_mlflow():
    global _MLFLOW_CFG
    global_cfg = get_global_config()

    if mlflow.active_run() is not None:
        return
    uri = resolve_tracking_uri()
    name = experiment_name(global_cfg.experiment_name)
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.set_experiment(name)
    run = mlflow.start_run(run_name=get_run_name(), tags=standard_tags(global_cfg.experiment_name))

    _MLFLOW_CFG = MLFlowConfig(
        uri=uri,
        experiment_name=name,
        experiment=experiment,
        run_id=run.info.run_id,
        run_name=run.info.run_name,
        active_run=run
    )
    print(
        f"MLflow: {uri}; experimento={experiment.name}; "
        f"run_id={run.info.run_id}"
    )


RUN_ID_FILE = "mlflow_run_id.txt"


def link_training_run(output_dir: Path, resumed: bool) -> str | None:
    """Record the active run id in ``output_dir``; on resume, carry over the previous run's history.

    MLflow params are immutable, so a resumed training logs into a new run instead of reopening
    the old one. The new run is tagged ``resumed_from_run_id`` and gets a copy of every metric
    history of the previous run, so the curves stay complete in a single run.
    """
    run = mlflow.active_run()
    if run is None:
        return None
    path = Path(output_dir) / RUN_ID_FILE
    previous = path.read_text(encoding="utf-8").strip() if resumed and path.is_file() else None
    if previous and previous != run.info.run_id:
        client = mlflow.MlflowClient()
        try:
            old = client.get_run(previous)
        except Exception as error:  # noqa: BLE001 - the previous run may live in another store
            warnings.warn(f"No se pudo leer la run previa {previous}: {error}", stacklevel=2)
        else:
            from mlflow.entities import Metric

            metrics = [
                Metric(m.key, m.value, m.timestamp, m.step)
                for key in old.data.metrics
                for m in client.get_metric_history(previous, key)
            ]
            for start in range(0, len(metrics), 1000):
                client.log_batch(run.info.run_id, metrics=metrics[start:start + 1000])
            mlflow.set_tag("resumed_from_run_id", previous)
            client.set_tag(previous, "resumed_in_run_id", run.info.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.info.run_id, encoding="utf-8")
    return previous


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


def set_run_name(name: str) -> None:
    mlflow.set_tag("mlflow.runName", name)


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
        experiment_name=experiment_name(),
        run_name=run.info.run_name,
        run_id=run.info.run_id,
        tracking_uri=resolve_tracking_uri(),
        save_dir=str(_MLFLOW_DIR),
    )


def attach_mlflow(trainer) -> None:
    """Add the run's Lightning logger and the shared ``val/tile/...`` metric keys to an RF-DETR trainer."""
    from .standard_metrics import StandardValMetrics

    trainer.loggers = [*trainer.loggers, get_mlflow_logger()]
    trainer.callbacks.append(StandardValMetrics())


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
