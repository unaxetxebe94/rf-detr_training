import pytest

from src.pipelines.disk import mlflow as tracking


@pytest.fixture(autouse=True)
def isolated_tracking(tmp_path, monkeypatch):
    """Tests never write to the central MLflow server: every test gets its own file store."""
    import mlflow

    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", (tmp_path / "isolated-tracking").as_uri())
    try:
        yield
    finally:
        tracking.stop_mlflow()
        mlflow.set_tracking_uri(previous_uri)
