"""Prepare the cached pipeline: update the incremental tile cache, build the hardlinked view and verify it.

Usage: ``uv run python -m src.pipelines.cached.prepare [--full] [--deep-verify]``.
"""

import argparse
import logging
import sys
from contextlib import ExitStack
from pathlib import Path

from ..disk.params import get_cached_config, get_data_config, get_global_config, get_preprocess_config
from .cache import update_cache
from .lock import CacheLock
from .verify import verify_view
from .view import DatasetView, build_view, load_view

logger = logging.getLogger(__name__)


def locks(data_cfg, cached_cfg) -> ExitStack:
    """Locks for the view directory and (if elsewhere) the cache; release them by closing the stack."""
    stack = ExitStack()
    dataset_dir = Path(data_cfg.training_dataset_path).resolve()
    cache_dir = cached_cfg.resolved_cache_dir(data_cfg.training_dataset_path).resolve()
    try:
        stack.enter_context(CacheLock(dataset_dir))
        if cache_dir != dataset_dir:
            stack.enter_context(CacheLock(cache_dir))
    except BaseException:
        stack.close()
        raise
    return stack


def prepare(*, full: bool = False, skip_update: bool = False, deep_verify: bool = False) -> tuple[DatasetView, dict]:
    """Bring the view up to date (or reuse it with ``skip_update``) and verify its files. Caller holds the locks."""
    data_cfg = get_data_config()
    preprocess_cfg = get_preprocess_config()
    cached_cfg = get_cached_config()
    dataset_dir = Path(data_cfg.training_dataset_path)
    stats: dict = {}
    if skip_update:
        view = load_view(dataset_dir)
        logger.info(f"Se reutiliza la vista existente de {dataset_dir} (fingerprint {view.fingerprint[:12]}).")
    else:
        results = update_cache(
            data_cfg.sources, cached_cfg.resolved_cache_dir(dataset_dir), preprocess_cfg, get_global_config(),
            cached_cfg, full=full,
        )
        for result in results:
            for key, value in result.stats.items():
                stats[f"cache_{key}"] = stats.get(f"cache_{key}", 0) + value
        view, view_stats = build_view(
            results, dataset_dir, preprocess_cfg.classes, cached_cfg.tile_format, n_jobs=preprocess_cfg.n_jobs
        )
        stats.update({f"view_{key}": value for key, value in view_stats.items()})
    checked = verify_view(dataset_dir, deep=deep_verify, n_jobs=preprocess_cfg.n_jobs)
    logger.info(f"Vista verificada ({'profunda' if deep_verify else 'rápida'}): {checked} ficheros.")
    print(f"Dataset: {view.summary}")
    return view, stats


def add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--full", action="store_true", help="Regenera todos los tiles de la caché.")
    parser.add_argument("--skip-update", action="store_true",
                        help="No actualiza la caché: reutiliza la vista ya construida.")
    parser.add_argument("--deep-verify", action="store_true", help="Decodifica todos los tiles antes de entrenar.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Actualiza la caché de tiles y la vista del pipeline cached.")
    add_prepare_arguments(parser)
    args = parser.parse_args(argv)
    if args.full and args.skip_update:
        parser.error("--full y --skip-update son incompatibles.")
    logging.basicConfig(level=logging.INFO)
    with locks(get_data_config(), get_cached_config()):
        prepare(full=args.full, skip_update=args.skip_update, deep_verify=args.deep_verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
