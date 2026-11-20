import pytest

from src.pipelines.disk import mlflow as tracking
from src.pipelines.disk.standard_metrics import standard_val_metrics
from src.test.metrics import per_class_detection_metrics, presence_metrics
from src.test.reporting import per_class_rows, standard_metrics

BOX = [0, 0, 10, 10]
FAR = [100, 100, 10, 10]
NAMES = {1: "short lipping", 2: "Water Mark"}


def test_presence_metrics_overall_and_one_vs_rest_per_class():
    truth = {"a": [(BOX, 1)], "b": [(BOX, 2)], "c": [], "d": []}
    predicted = {"a": [(BOX, 1, 0.9)], "b": [(BOX, 1, 0.8)], "c": [(BOX, 2, 0.7)], "d": []}
    scores = {**predicted, "d": [(BOX, 2, 0.1)]}

    metrics = presence_metrics(truth, predicted, NAMES, scores)

    overall = metrics["overall"]
    assert (overall["Total TP"], overall["Total FP"], overall["Total FN"], overall["Total TN"]) == (2, 1, 0, 1)
    short = metrics["short lipping"]  # "b" has no short lipping but was predicted as one
    assert (short["Total TP"], short["Total FP"], short["Total FN"], short["Total TN"]) == (1, 1, 0, 2)
    water = metrics["Water Mark"]
    assert (water["Total TP"], water["Total FP"], water["Total FN"], water["Total TN"]) == (0, 1, 1, 2)
    assert overall["AUROC"] == 1.0  # defective doors score 0.9/0.8 vs normal 0.7/0.1
    assert "AUROC" in short and "AP" in water


def test_presence_metrics_skips_scores_when_only_one_class_present():
    metrics = presence_metrics({"a": [], "b": []}, {"a": [], "b": [(BOX, 1, 0.9)]}, NAMES)
    assert "AUROC" not in metrics["overall"]
    assert metrics["overall"]["Total FP"] == 1


def test_per_class_detection_metrics_isolates_each_class():
    truth = {"a": [(BOX, 1), (FAR, 2)]}
    predictions = [("a", BOX, 1, 0.9), ("a", BOX, 2, 0.8)]

    metrics = per_class_detection_metrics(predictions, truth, NAMES)

    assert metrics["short lipping"]["Total TP"] == 1
    assert metrics["short lipping"]["mAP@50"] == 1.0
    assert (metrics["Water Mark"]["Total TP"], metrics["Water Mark"]["Total FP"]) == (0, 1)


def test_standard_metric_keys_for_tile_and_image_levels():
    presence = {"overall": {"F1-Score": 0.5, "Total TP": 1}, "Water Mark": {"Recall": 1.0}}
    detection = {"Water Mark": {"mAP@50": 0.25}}
    results = {
        "F1-Score": 0.4, "mAP@50": 0.3,
        "tile_presence": presence, "tile_detection_per_class": detection,
        "image_presence": presence, "image_detection_per_class": detection,
        "image_detection_metrics": {"Recall": 0.9},
        "per_class_thresholds": {"Water Mark": 0.35},
    }

    metrics = standard_metrics(results)

    assert metrics["test/tile/f1"] == 0.5
    assert metrics["test/tile/tp"] == 1
    assert metrics["test/tile/detection/f1"] == 0.4
    assert metrics["test/tile/detection/map50"] == 0.3
    assert metrics["test/tile/class/water_mark/recall"] == 1.0
    assert metrics["test/tile/class/water_mark/detection/map50"] == 0.25
    assert metrics["test/tile/class/water_mark/threshold"] == 0.35
    assert metrics["test/image/f1"] == 0.5
    assert metrics["test/image/detection/recall"] == 0.9
    assert metrics["test/image/class/water_mark/recall"] == 1.0
    assert {row["level"] for row in per_class_rows(results)} == {"tile", "image"}


def test_validation_metrics_are_mirrored_to_standard_keys():
    metrics = standard_val_metrics({
        "val/mAP_50": 0.5, "val/F1": 0.4, "val/AP/Short Lipping": 0.3, "val/loss": 1.0,
        "val/mAR": float("nan"),
    })
    assert metrics == {
        "val/tile/detection/map50": 0.5,
        "val/tile/detection/f1": 0.4,
        "val/tile/class/short_lipping/detection/map50_95": 0.3,
    }


def test_unreachable_server_falls_back_to_local_store(monkeypatch):
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", None)
    monkeypatch.setattr(tracking, "_TRACKING_FALLBACK", False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
    with pytest.warns(UserWarning, match="no responde"):
        uri = tracking.resolve_tracking_uri()
    assert uri == tracking._MLFLOW_DIR.as_uri()
    assert tracking.standard_tags("top-bottom")["tracking_fallback"] == "True"


def test_experiments_are_grouped_by_model_family():
    assert tracking.experiment_name("top-bottom") == "rfdetr/top-bottom"
    assert tracking.experiment_name("rfdetr/top-bottom") == "rfdetr/top-bottom"
