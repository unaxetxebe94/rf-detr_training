"""Compare trainings side by side: dataset, preprocessing, parameters, environment and metrics.

Usage: ``uv run python -m src.pipelines.cached.compare [--last 5 | --runs ID ...] [--all] [--csv FILE]``.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from ..disk.mlflow import MLFLOW_TRACKING_URI
from ..disk.params import get_global_config

import mlflow  # noqa: E402  (after ..disk.mlflow, which allows MLflow's file store)
from mlflow import MlflowClient  # noqa: E402

DEFAULT_METRICS = r"^(tile|image|test)/"
SECTION_ORDER = ("metric", "tag", "dataset", "delta", "preprocess", "cached", "train", "global", "env")


def select_runs(client: MlflowClient, experiment: str, run_ids: list[str] | None, last: int):
    if run_ids:
        return [client.get_run(run_id) for run_id in run_ids]
    found = client.get_experiment_by_name(experiment)
    if found is None:
        raise ValueError(f"No existe el experimento de MLflow '{experiment}'.")
    runs = client.search_runs(
        [found.experiment_id], "tags.stage_train = 'True'", order_by=["attributes.start_time DESC"], max_results=last
    )
    return runs[::-1]


def run_column(run) -> str:
    return f"{run.info.run_name or run.info.run_id[:8]} ({run.info.run_id[:6]})"


def build_table(runs, metrics_pattern: str = DEFAULT_METRICS, only_differences: bool = True) -> pd.DataFrame:
    """Rows are fields (``section | key``), columns are runs; numeric rows get a ``Δ último-primero`` column."""
    metric_filter = re.compile(metrics_pattern)
    columns = {}
    for run in runs:
        values = {f"metric | {key}": value for key, value in run.data.metrics.items() if metric_filter.search(key)}
        values.update({f"tag | {key}": value for key, value in run.data.tags.items() if not key.startswith("mlflow.")})
        for key, value in run.data.params.items():
            section = key.split("/", 1)[0] if "/" in key else "global"
            values[f"{section} | {key}"] = value
        columns[run_column(run)] = values
    table = pd.DataFrame(columns)
    if only_differences and len(runs) > 1:
        table = table[table.apply(lambda row: row.astype(str).nunique(dropna=False) > 1, axis=1)]
    order = {section: index for index, section in enumerate(SECTION_ORDER)}
    table = table.loc[sorted(table.index, key=lambda name: (order.get(name.split(" | ")[0], len(order)), name))]
    if len(runs) > 1:
        numbers = table.apply(pd.to_numeric, errors="coerce")
        first, last = numbers.iloc[:, 0], numbers.iloc[:, -1]
        table["Δ último-primero"] = (last - first).round(6)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compara runs de entrenamiento de MLflow.")
    parser.add_argument("--runs", nargs="+", help="IDs de run a comparar (por defecto, los últimos --last).")
    parser.add_argument("--last", type=int, default=5, help="Número de entrenamientos recientes a comparar.")
    parser.add_argument("--experiment", help="Experimento de MLflow (por defecto, el de params.yaml).")
    parser.add_argument("--metrics", default=DEFAULT_METRICS, help="Regex de las métricas a mostrar.")
    parser.add_argument("--all", action="store_true", help="Muestra también los campos idénticos en todas las runs.")
    parser.add_argument("--csv", type=Path, help="Guarda la tabla en este CSV.")
    args = parser.parse_args(argv)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = args.experiment or get_global_config().experiment_name
    runs = select_runs(client, experiment, args.runs, args.last)
    if not runs:
        print("No hay entrenamientos que comparar.")
        return 1
    table = build_table(runs, args.metrics, only_differences=not args.all)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 250,
                           "display.max_colwidth", 60):
        print(table.fillna("-").to_string())
    if args.csv:
        table.to_csv(args.csv, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    sys.exit(main())
