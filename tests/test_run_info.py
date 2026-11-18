from types import SimpleNamespace

import pytest
from test_cached import IMAGES, TILES_PER_IMAGE, run, source  # noqa: F401

import mlflow
from mlflow import MlflowClient
from src.pipelines.cached import run_info
from src.pipelines.cached.compare import build_table
from src.pipelines.cached.train import check_init_categories, resolve_init
from src.pipelines.disk import mlflow as tracking


def test_dataset_report_counts_per_class_and_split(source, tmp_path):  # noqa: F811
    _, view = run(source, tmp_path)
    flat, splits = run_info.dataset_report(view.dataset_dir, view.categories)
    assert flat["all/originals"] == IMAGES
    assert flat["all/tiles"] == IMAGES * TILES_PER_IMAGE
    assert sum(len(images) for images in splits.values()) == IMAGES
    for cat in view.categories:
        per_split = sum(flat[f"{split}/class/{cat['name']}/annotations"] for split in splits)
        assert per_split <= flat["all/annotations"]
    assert flat["all/annotations"] == sum(
        value for key, value in flat.items() if key.endswith("/annotations") and "/class/" in key
    )
    assert run_info.epoch_size(view.summary, 0.0) == view.summary["train/positive_tiles"]


@pytest.fixture
def store(tmp_path, monkeypatch):
    previous_uri = mlflow.get_tracking_uri()
    uri = (tmp_path / "tracking").as_uri()
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(tracking, "get_global_config", lambda: SimpleNamespace(experiment_name="right"))
    monkeypatch.setattr(tracking, "get_run_name", lambda: "run")
    try:
        yield MlflowClient(tracking_uri=uri)
    finally:
        tracking.stop_mlflow()
        mlflow.set_tracking_uri(previous_uri)


def test_lineage_links_parent_and_logs_dataset_deltas(store, tmp_path):
    checkpoint = tmp_path / "first" / "checkpoint_best_total.pth"
    with tracking.mlflow_run("train"):
        tracking.log_dict_params({"fingerprint": "a", "all/annotations": 10, "train/class/dent/annotations": 4},
                                 prefix="dataset/")
        tracking.log_dict_params({"training/checkpoint": str(checkpoint.resolve())})
        first = mlflow.active_run().info.run_id
    with tracking.mlflow_run("train"):
        deltas = run_info.register_lineage(
            checkpoint, {"fingerprint": "b", "all/annotations": 25, "train/class/dent/annotations": 9},
        )
        second = mlflow.active_run().info.run_id
        tracking.log_dict_params({"all/annotations": 25}, prefix="dataset/")
    assert deltas == {"all/annotations": 15, "train/class/dent/annotations": 5}
    saved = store.get_run(second)
    assert saved.data.tags["parent_run_id"] == first
    assert saved.data.tags["warm_start"] == "True"
    assert saved.data.tags["same_dataset_as_parent"] == "False"
    table = build_table([store.get_run(first), saved], only_differences=True)
    assert table.loc["dataset | dataset/all/annotations", "Δ último-primero"] == 15


def test_init_checkpoint_requires_same_categories(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    (training / "checkpoint_best_total.pth").touch()
    categories = [{"id": 1, "name": "dent"}]
    (training / "training_categories.json").write_text('{"categories": [{"id": 1, "name": "dent"}]}')
    checkpoint = resolve_init(training)
    assert checkpoint == training / "checkpoint_best_total.pth"
    check_init_categories(checkpoint, categories)
    with pytest.raises(ValueError, match="no coinciden"):
        check_init_categories(checkpoint, [*categories, {"id": 2, "name": "scratch"}])
    with pytest.raises(FileNotFoundError):
        resolve_init(tmp_path / "missing")
