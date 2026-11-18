"""Everything needed to compare trainings later: dataset composition, environment, lineage and artifacts."""

import hashlib
import json
import platform
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from importlib import metadata
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from ...preprocess.annotations import SPLITS, read_annotations
from ..disk.mlflow import _sanitize_key
from .view import ANNOTATIONS_FILE

PACKAGES = ("rfdetr", "torch", "pytorch-lightning", "kornia", "albumentations", "mlflow", "numpy", "pyvips")
RUN_INFO_DIR = "run_info"


def dataset_report(dataset_dir: Path, categories: list[dict]) -> tuple[dict, dict]:
    """Flat per-split statistics (counts per class included) and the original images of every split."""
    names = {cat["id"]: str(cat["name"]) for cat in categories}
    flat: dict = {}
    splits: dict[str, list[str]] = {}
    totals: dict[str, int] = defaultdict(int)
    for split in SPLITS:
        coco = read_annotations(Path(dataset_dir) / split / ANNOTATIONS_FILE)
        originals = sorted({image["source_image_id"] for image in coco["images"]})
        splits[split] = originals
        flat[f"{split}/originals_hash"] = hashlib.sha256("\n".join(originals).encode("utf-8")).hexdigest()[:16]
        by_image = defaultdict(set)
        per_class_annotations: dict[str, int] = defaultdict(int)
        for ann in coco["annotations"]:
            name = names.get(ann["category_id"], str(ann["category_id"]))
            per_class_annotations[name] += 1
            by_image[name].add(ann["image_id"])
        for name in names.values():
            flat[f"{split}/class/{name}/annotations"] = per_class_annotations.get(name, 0)
            flat[f"{split}/class/{name}/tiles"] = len(by_image.get(name, ()))
        totals["originals"] += len(originals)
        totals["annotations"] += len(coco["annotations"])
        totals["tiles"] += len(coco["images"])
    flat.update({f"all/{key}": value for key, value in totals.items()})
    return flat, splits


def epoch_size(summary: dict, negative_ratio: float) -> int:
    """Tiles per epoch before padding: all positives plus the sampled share of negatives."""
    return summary["train/positive_tiles"] + round(negative_ratio * summary["train/negative_tiles"])


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def environment_info() -> dict:
    """Code version, library versions and hardware, to explain differences between trainings."""
    info: dict = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in PACKAGES:
        try:
            info[f"version_{package}"] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {key: value for key, value in info.items() if value is not None}


def _numeric(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_parent_run(init_checkpoint: Path | None, current_run_id: str, experiment_id: str):
    """Run that produced ``init_checkpoint`` or, without warm start, the latest finished training."""
    client = MlflowClient()
    filters = ["tags.stage_train = 'True'", "attributes.status = 'FINISHED'"]
    if init_checkpoint is not None:
        filters.append(f"params.`training/checkpoint` = '{Path(init_checkpoint).resolve()}'")
    runs = client.search_runs(
        [experiment_id], " and ".join(filters), order_by=["attributes.start_time DESC"], max_results=10
    )
    return next((run for run in runs if run.info.run_id != current_run_id), None)


def register_lineage(init_checkpoint: Path | None, dataset_params: dict) -> dict:
    """Tag the active run with its parent and log how the dataset changed since then (``delta/*`` params)."""
    run = mlflow.active_run()
    tags = {"warm_start": init_checkpoint is not None}
    if init_checkpoint is not None:
        tags["init_checkpoint"] = str(Path(init_checkpoint).resolve())
    parent = find_parent_run(init_checkpoint, run.info.run_id, run.info.experiment_id)
    deltas = {}
    if parent is not None:
        tags["parent_run_id"] = parent.info.run_id
        tags["parent_run_name"] = parent.info.run_name
        parent_params = parent.data.params
        for key, value in dataset_params.items():
            current = _numeric(value)
            previous = _numeric(parent_params.get(f"dataset/{_sanitize_key(key)}"))
            if current is not None and previous is not None and not isinstance(value, bool):
                deltas[key] = current - previous
        parent_fingerprint = parent_params.get("dataset/fingerprint")
        if parent_fingerprint:
            tags["same_dataset_as_parent"] = parent_fingerprint == dataset_params.get("fingerprint")
    mlflow.set_tags({key: str(value) for key, value in tags.items()})
    return {key: int(value) if float(value).is_integer() else value for key, value in deltas.items()}


def run_label(dataset_params: dict, warm_start: bool) -> str:
    """Short, readable run name: when, how much data and which dataset."""
    return (
        f"{datetime.now():%Y-%m-%d %H:%M} | {dataset_params['all/originals']} img | "
        f"{dataset_params['all/annotations']} ann | {str(dataset_params['fingerprint'])[:7]}"
        f"{' | warm' if warm_start else ''}"
    )


def save_run_info(output_dir: Path, files: dict[str, dict], extra_files: tuple[Path, ...] = ()) -> None:
    """Write the JSONs (and copies of ``extra_files``) next to the training and attach them to the active run."""
    target = Path(output_dir) / RUN_INFO_DIR
    target.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (target / name).write_text(json.dumps(content, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for path in extra_files:
        if Path(path).is_file():
            shutil.copy2(path, target / Path(path).name)
    mlflow.log_artifacts(str(target), artifact_path=RUN_INFO_DIR)
