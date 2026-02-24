"""
process_results.py
------------------
Convierte los outputs del entrenamiento (log.txt y results.json) en ficheros
compatibles con `dvc metrics` y `dvc plots`.

Outputs
-------
trainings/temp/dvc_metrics.json   → dvc metrics show
trainings/temp/dvc_plots.json     → dvc plots show (curvas por epoch)
"""

import json
import yaml
import logging
from pathlib import Path
from logger import get_logger

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
# Main
# ---------------------------------------------------------------------------

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    task_name  = params["task-name"]
    train_dir  = Path("trainings", task_name)
    output_dir = Path("trainings", "temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Plots desde log.txt ───────────────────────────────────────────
    log_path = train_dir / "log.txt"
    if not log_path.exists():
        logger.error(f"No se encontró {log_path}")
        raise FileNotFoundError(log_path)

    plot_records = parse_log(log_path)

    plots_out = output_dir / "dvc_plots.json"
    with open(plots_out, "w", encoding="utf-8") as f:
        json.dump(plot_records, f, indent=2)
    logger.info(f"Plots escritos → {plots_out}")

    # ── 2. Métricas desde results.json ──────────────────────────────────
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


if __name__ == "__main__":
    main()