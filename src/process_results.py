"""
process_results.py
------------------
Convierte los outputs del entrenamiento (log.txt y results.json) en ficheros
compatibles con `dvc metrics` y `dvc plots`, y los sube a W&B como gráficas
y métricas de resumen.

Outputs
-------
trainings/temp/dvc_metrics.json   → dvc metrics show
trainings/temp/dvc_plots.json     → dvc plots show (curvas por epoch)
"""

import json
import yaml
import wandb
import logging
from pathlib import Path
from logger import get_logger
from utils import read_params

logger = get_logger(__name__, level=logging.DEBUG)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_get(d: dict, *keys, default=None):
    """Navega un dict anidado de forma segura."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def _extract_class_metrics(results_json: dict, class_name: str = "all") -> dict:
    """Extrae las métricas de una clase concreta desde el bloque results_json del log."""
    class_map = results_json.get("class_map", [])
    for entry in class_map:
        if entry.get("class") == class_name:
            return {
                "map@50:95": entry.get("map@50:95"),
                "map@50":    entry.get("map@50"),
                "precision": entry.get("precision"),
                "recall":    entry.get("recall"),
                "f1_score":  entry.get("f1_score"),
            }
    # Fallback: campos de primer nivel
    return {
        "map@50:95": results_json.get("map"),
        "map@50":    results_json.get("map"),
        "precision": results_json.get("precision"),
        "recall":    results_json.get("recall"),
        "f1_score":  results_json.get("f1_score"),
    }


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_log(log_path: Path) -> list[dict]:
    """
    Lee el log.txt (JSONL, una línea por epoch) y devuelve una lista de dicts
    aplanados con las métricas clave para plots.
    """
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"log.txt línea {lineno}: JSON inválido – {e}")
                continue

            epoch = raw.get("epoch", lineno - 1)

            # Métricas del modelo regular (test set)
            reg_res = raw.get("test_results_json", {})
            reg     = _extract_class_metrics(reg_res)

            # Métricas del modelo EMA (test set)
            ema_res = raw.get("ema_test_results_json", {})
            ema     = _extract_class_metrics(ema_res)

            record = {
                "epoch": epoch,

                # Losses
                "train_loss":     raw.get("train_loss"),
                "test_loss":      raw.get("test_loss"),
                "ema_test_loss":  raw.get("ema_test_loss"),

                # Component losses (útiles para diagnosticar)
                "train_loss_ce":   raw.get("train_loss_ce"),
                "train_loss_bbox": raw.get("train_loss_bbox"),
                "train_loss_giou": raw.get("train_loss_giou"),
                "test_loss_ce":    raw.get("test_loss_ce"),
                "test_loss_bbox":  raw.get("test_loss_bbox"),
                "test_loss_giou":  raw.get("test_loss_giou"),

                # Learning rate
                "lr": raw.get("train_lr"),

                # Métricas modelo regular
                "map50":     reg.get("map@50"),
                "map50_95":  reg.get("map@50:95"),
                "precision": reg.get("precision"),
                "recall":    reg.get("recall"),
                "f1":        reg.get("f1_score"),

                # Métricas modelo EMA
                "ema_map50":     ema.get("map@50"),
                "ema_map50_95":  ema.get("map@50:95"),
                "ema_precision": ema.get("precision"),
                "ema_recall":    ema.get("recall"),
                "ema_f1":        ema.get("f1_score"),

                # Mejor resultado acumulado hasta este epoch
                "best_map50_95": raw.get("all_best_res"),
                "best_epoch":    raw.get("all_best_ep"),
            }

            records.append(record)

    logger.info(f"log.txt: {len(records)} epochs parseados")
    return records


def parse_results(results_path: Path) -> dict:
    """
    Lee el results.json final y construye el dict de métricas para DVC.
    Soporta tanto el formato con splits valid/test como el formato plano.
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = {}

    class_map = data.get("class_map", {})

    # Formato con splits: {"class_map": {"valid": [...], "test": [...]}, ...}
    if isinstance(class_map, dict):
        for split in ("valid", "test"):
            entries = class_map.get(split, [])
            for entry in entries:
                cls = entry.get("class", "unknown")
                prefix = f"{split}_{cls}"
                metrics[f"{prefix}_map50_95"]  = entry.get("map@50:95")
                metrics[f"{prefix}_map50"]     = entry.get("map@50")
                metrics[f"{prefix}_precision"] = entry.get("precision")
                metrics[f"{prefix}_recall"]    = entry.get("recall")
                metrics[f"{prefix}_f1"]        = entry.get("f1_score")

    # Formato plano (como en los logs por epoch): {"class_map": [...], ...}
    elif isinstance(class_map, list):
        for entry in class_map:
            cls = entry.get("class", "unknown")
            metrics[f"{cls}_map50_95"]  = entry.get("map@50:95")
            metrics[f"{cls}_map50"]     = entry.get("map@50")
            metrics[f"{cls}_precision"] = entry.get("precision")
            metrics[f"{cls}_recall"]    = entry.get("recall")
            metrics[f"{cls}_f1"]        = entry.get("f1_score")

    # Campos de primer nivel (resumen global)
    for key in ("map", "precision", "recall", "f1_score"):
        if key in data:
            metrics[key] = data[key]

    logger.info(f"results.json: {len(metrics)} métricas extraídas")
    return metrics


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------

def _connect_wandb(params: dict):
    """
    Intenta retomar el run de W&B creado durante el entrenamiento.
    Devuelve el run activo o None si no se puede conectar.
    """
    run_name_path = Path("trainings", "temp", "run_info.json")
    if not run_name_path.exists():
        logger.warning("No se encontró run_info.json — no se logeará en W&B.")
        return None

    with open(run_name_path) as f:
        run_name = json.load(f)["run_name"]

    try:
        api = wandb.Api()
        runs = api.runs(
            f"{params['train']['entity']}/{params['train']['project']}"
            if "entity" in params.get("train", {})
            else params["train"]["project"],
            filters={"display_name": run_name},
        )
        if runs.length > 0:
            run_id = runs[0].id
            run = wandb.init(
                project=params["train"]["project"],
                id=run_id,
                resume="must",
                job_type="process_results",
            )
            logger.info(f"Conectado al run W&B '{run_name}' (id: {run_id})")
            return run
        else:
            logger.warning(f"No se encontró el run '{run_name}' en W&B.")
            return None
    except Exception as e:
        logger.error(f"Error conectando a W&B: {e}")
        return None


def log_curves_to_wandb(run, plot_records: list[dict]) -> None:
    """
    Sube las curvas de entrenamiento epoch a epoch a W&B usando tablas
    personalizadas, separadas en grupos temáticos.
    """
    if not plot_records:
        logger.warning("No hay registros de epochs para logar en W&B.")
        return

    def _series_from_records(records, x_key: str, y_keys: list[str]):
        """Devuelve xs, ys listas aptas para wandb.plot.line_series."""
        xs = [r.get(x_key) for r in records]
        ys = [[r.get(k) for r in records] for k in y_keys]
        return xs, ys

    # ── Curvas de Loss ────────────────────────────────────────────────────────
    loss_keys = ["train_loss", "test_loss", "ema_test_loss"]
    xs, ys = _series_from_records(plot_records, "epoch", loss_keys)
    try:
        run.log({
            "curves/losses":
                wandb.plot.line_series(
                    xs=xs,
                    ys=ys,
                    keys=loss_keys,
                    title="Losses por epoch",
                    xname="epoch",
                )
        })
    except Exception as e:
        logger.error(f"Error subiendo curvas de losses a W&B: {e}")

    # ── Loss desglosado (CE / BBox / GIoU) ───────────────────────────────────
    # loss_detail_keys = [
    #     "train_loss_ce", "train_loss_bbox", "train_loss_giou",
    #     "test_loss_ce",  "test_loss_bbox",  "test_loss_giou",
    # ]
    # xs, ys = _series_from_records(plot_records, "epoch", loss_detail_keys)
    # try:
    #     run.log({
    #         "curves/loss_components":
    #             wandb.plot.line_series(
    #                 xs=xs,
    #                 ys=ys,
    #                 keys=loss_detail_keys,
    #                 title="Componentes de loss por epoch",
    #                 xname="epoch",
    #             )
    #     })
    # except Exception as e:
    #     logger.error(f"Error subiendo componentes de loss a W&B: {e}")

    # ── mAP ──────────────────────────────────────────────────────────────────
    map_keys = ["map50", "map50_95", "ema_map50", "ema_map50_95", "best_map50_95"]
    xs, ys = _series_from_records(plot_records, "epoch", map_keys)
    try:
        run.log({
            "curves/mAP":
                wandb.plot.line_series(
                    xs=xs,
                    ys=ys,
                    keys=map_keys,
                    title="mAP por epoch",
                    xname="epoch",
                )
        })
    except Exception as e:
        logger.error(f"Error subiendo mAP a W&B: {e}")

    # ── Precision / Recall / F1 ───────────────────────────────────────────────
    prf_keys = [
        # "precision", "recall", "f1",
        "ema_precision", "ema_recall", "ema_f1",
    ]
    xs, ys = _series_from_records(plot_records, "epoch", prf_keys)
    try:
        run.log({
            "curves/precision_recall_f1":
                wandb.plot.line_series(
                    xs=xs,
                    ys=ys,
                    keys=prf_keys,
                    title="Precision / Recall / F1 por epoch",
                    xname="epoch",
                )
        })
    except Exception as e:
        logger.error(f"Error subiendo Precision/Recall/F1 a W&B: {e}")

    # ── Learning rate ─────────────────────────────────────────────────────────
    lr_keys = ["lr"]
    xs, ys = _series_from_records(plot_records, "epoch", lr_keys)
    try:
        run.log({
            "curves/learning_rate":
                wandb.plot.line_series(
                    xs=xs,
                    ys=ys,
                    keys=lr_keys,
                    title="Learning rate por epoch",
                    xname="epoch",
                )
        })
    except Exception as e:
        logger.error(f"Error subiendo learning rate a W&B: {e}")

    logger.info("Curvas de entrenamiento subidas a W&B.")


def log_metrics_to_wandb(run, metrics: dict) -> None:
    """Sube las métricas finales al summary del run de W&B."""
    if not metrics:
        return
    for k, v in metrics.items():
        if v is not None:
            run.summary[f"final/{k}"] = v
    logger.info(f"Métricas finales ({len(metrics)}) subidas al summary de W&B.")


def log_preprocess_info_to_wandb(run) -> None:
    """
    Si existe trainings/temp/preprocess_info.json, sube los parámetros y
    estadísticas del preprocesado como config y tabla en W&B.
    """
    preprocess_info_path = Path("trainings", "temp", "preprocess_info.json")
    if not preprocess_info_path.exists():
        logger.info("No hay preprocess_info.json — se omite el logging de preprocesado.")
        return

    with open(preprocess_info_path) as f:
        info = json.load(f)

    # Parámetros de preprocesado → config del run (para comparar entre runs)
    if "params" in info:
        for k, v in info["params"].items():
            run.config[f"preprocess/{k}"] = v

    # Estadísticas de splits → tabla W&B
    if "split_stats" in info:
        stats = info["split_stats"]
        table = wandb.Table(columns=["split", "n_images", "n_annotations"])
        for split, data in stats.items():
            table.add_data(split, data.get("n_images", 0), data.get("n_annotations", 0))
        run.log({"preprocess/split_stats": table})

    logger.info("Información de preprocesado subida a W&B.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    params = read_params()

    task_name  = params["task-name"]
    train_dir  = Path("trainings", "training")
    output_dir = Path("trainings", "temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Plots desde log.txt ───────────────────────────────────────────────
    log_path = train_dir / "log.txt"
    if not log_path.exists():
        logger.error(f"No se encontró {log_path}")
        raise FileNotFoundError(log_path)

    plot_records = parse_log(log_path)

    plots_out = output_dir / "dvc_plots.json"
    with open(plots_out, "w", encoding="utf-8") as f:
        json.dump(plot_records, f, indent=2)
    logger.info(f"Plots escritos → {plots_out}")

    # ── 2. Métricas desde results.json ──────────────────────────────────────
    results_path = train_dir / "results.json"
    if not results_path.exists():
        logger.warning(
            f"No se encontró {results_path}. "
            "Usando métricas del último epoch del log."
        )
        if plot_records:
            last = plot_records[-1]
            metrics = {
                "map50":     last.get("map50"),
                "map50_95":  last.get("map50_95"),
                "precision": last.get("precision"),
                "recall":    last.get("recall"),
                "f1":        last.get("f1"),
                "ema_map50":    last.get("ema_map50"),
                "ema_map50_95": last.get("ema_map50_95"),
                "ema_f1":       last.get("ema_f1"),
                "best_map50_95": last.get("best_map50_95"),
                "best_epoch":    last.get("best_epoch"),
            }
        else:
            metrics = {}
    else:
        metrics = parse_results(results_path)

    metrics_out = output_dir / "dvc_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Métricas escritas → {metrics_out}")

    # ── 3. Logging a W&B ────────────────────────────────────────────────────
    run = _connect_wandb(params)
    if run is not None:
        log_curves_to_wandb(run, plot_records)
        log_metrics_to_wandb(run, metrics)
        log_preprocess_info_to_wandb(run)
        wandb.finish()
        logger.info("W&B: logging completado y run cerrado.")
    else:
        logger.warning("W&B no disponible — sólo se han escrito los ficheros DVC.")


if __name__ == "__main__":
    main()