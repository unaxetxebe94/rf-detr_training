from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import mlflow
from mlflow import MlflowClient
from pipelines.disk import mlflow as tracking
from pipelines.disk import train
from pipelines.disk.__main__ import main as pipeline
from pipelines.disk.params import GlobalConfig, PreprocessConfig, TrainConfig
from src.test import run_test as evaluation
from src.test.reporting import log_results_to_mlflow


@pytest.fixture
def store(tmp_path, monkeypatch):
    previous_uri = mlflow.get_tracking_uri()
    uri = (tmp_path / "tracking").as_uri()
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(tracking, "get_global_config", lambda: SimpleNamespace(experiment_name="right"))
    monkeypatch.setattr(tracking, "get_run_name", lambda: "regression")
    try:
        yield MlflowClient(tracking_uri=uri)
    finally:
        tracking.stop_mlflow()
        mlflow.set_tracking_uri(previous_uri)


def test_logger_and_evaluation_share_persistent_run(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with tracking.mlflow_run("pipeline") as run:
        logger = tracking.get_mlflow_logger()
        assert logger.run_id == run.info.run_id
        assert logger._tracking_uri == tracking.MLFLOW_TRACKING_URI
        logger.log_metrics({"train/loss": 1.25}, step=1)
        with tracking.mlflow_run("test"):
            assert log_results_to_mlflow({
                "Total TP": 2,
                "evaluation_scope": "tiles",
                "door_evaluation_skip_reason": "No complete images",
                "threshold_calibration": {"global_threshold": 0.5},
            }) == run.info.run_id
        assert mlflow.active_run().info.run_id == run.info.run_id
    experiment = store.get_experiment_by_name("right")
    runs = store.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    saved = runs[0]
    assert saved.info.status == "FINISHED"
    assert saved.data.metrics == {"train/loss": 1.25, "tile/Total TP": 2.0}
    assert saved.data.params["evaluation_scope"] == "tiles"
    assert saved.data.params["calibration/global_threshold"] == "0.5"
    assert tracking.get_mlflow_cfg() is None


@pytest.mark.parametrize("fails", [False, True])
def test_independent_stage_closes_its_run(store, fails):
    def execute():
        with tracking.mlflow_run("train") as run:
            tracking.log_dict_params({"batch_size": 4})
            if fails:
                raise ValueError("training failed")
            return run.info.run_id

    if fails:
        with pytest.raises(ValueError, match="training failed"):
            execute()
    else:
        execute()
    assert mlflow.active_run() is None
    experiment = store.get_experiment_by_name("right")
    runs = store.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == ("FAILED" if fails else "FINISHED")


@pytest.mark.parametrize("fails", [False, True])
def test_pipeline_uses_its_checkpoint_and_records_failure(store, fails):
    checkpoint = Path("trainings") / "this-run" / "checkpoint_best_total.pth"
    with patch("src.core.__main__.train", return_value=checkpoint):
        with patch("src.core.__main__.run_test") as evaluate:
            if fails:
                evaluate.side_effect = ValueError("evaluation failed")
                with pytest.raises(ValueError, match="evaluation failed"):
                    pipeline()
            else:
                pipeline()
            evaluate.assert_called_once_with(["--weights", str(checkpoint)])
    experiment = store.get_experiment_by_name("right")
    runs = store.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == ("FAILED" if fails else "FINISHED")


def test_training_saves_categories_and_returns_its_checkpoint(store, tmp_path, monkeypatch):
    import json

    dataset = tmp_path / "dataset"
    annotations = dataset / "train" / "_annotations.coco.json"
    annotations.parent.mkdir(parents=True)
    categories = [{"id": 1, "name": "defect"}]
    annotations.write_text(json.dumps({"categories": categories}), encoding="utf-8")
    monkeypatch.setattr(train, "get_global_config", lambda: GlobalConfig("right", "large", 42))
    monkeypatch.setattr(train, "get_train_config", lambda: TrainConfig(0.0001, 0.00015, 4, 6, 3, 0.0001, False))
    monkeypatch.setattr(
        train, "get_preprocess_config", lambda: PreprocessConfig(1.0, 0.1, 0.33, True, 0.7, 0.15, 0.15, 2, 5),
    )
    monkeypatch.setattr(train, "get_data_config", lambda: SimpleNamespace(training_dataset_path=dataset))
    monkeypatch.setattr(train, "MODEL_CONFIGS", {"large": Mock()})
    monkeypatch.setattr(train, "RFDETRModelModule", Mock())
    monkeypatch.setattr(train, "RFDETRDataModule", Mock())
    output = tmp_path / "training"
    trainer = SimpleNamespace(loggers=[])
    trainer.fit = Mock(side_effect=lambda *args: (output / "checkpoint_best_total.pth").touch())
    monkeypatch.setattr(train, "build_trainer", Mock(return_value=trainer))
    checkpoint = train.main(output)
    assert checkpoint == output / "checkpoint_best_total.pth"
    assert json.loads((output / "training_categories.json").read_text()) == {"categories": categories}
    saved = store.search_runs([store.get_experiment_by_name("right").experiment_id])[0]
    assert saved.data.params["training/checkpoint"] == str(checkpoint.resolve())
    assert saved.info.status == "FINISHED"


def test_test_entry_closes_failed_run(store, tmp_path, monkeypatch):
    weights = tmp_path / "weights.pth"
    weights.touch()
    monkeypatch.setattr(evaluation, "get_global_config", lambda: SimpleNamespace(model_type="large"))
    monkeypatch.setattr(evaluation, "get_data_config", lambda: SimpleNamespace(
        training_dataset_path=tmp_path, sources=[],
    ))
    with patch("rfdetr.RFDETRLarge", side_effect=ValueError("invalid checkpoint")):
        with pytest.raises(ValueError, match="invalid checkpoint"):
            evaluation.run_test(["--weights", str(weights)])
    saved = store.search_runs([store.get_experiment_by_name("right").experiment_id])
    assert len(saved) == 1
    assert saved[0].info.status == "FAILED"
    assert mlflow.active_run() is None
