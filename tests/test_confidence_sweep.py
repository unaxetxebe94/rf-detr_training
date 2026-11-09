import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from src.preprocess.annotations import read_annotations, write_annotations
from src.scripts.plot_confidence_tradeoff import build_report, build_reports, main, save_report
from src.test.confidence_sweep import sweep_confidence_thresholds
from src.test.image_evaluation import reconstruct_predictions
from src.test.metrics import detection_metrics
from src.test.predictions import filter_predictions


def test_sweep_keeps_class_differences_and_includes_exact_endpoints_and_base():
    rows = sweep_confidence_thresholds({}, {}, {1: 0.23, 2: 0.71}, step=0.1)
    assert [row["offset"] for row in rows] == [-0.23, -0.2, -0.1, 0, 0.1, 0.2, 0.29]
    assert rows[0]["thresholds"] == {1: 0.0, 2: 0.48}
    assert rows[-1]["thresholds"] == {1: 0.52, 2: 1.0}
    assert next(row for row in rows if row["offset"] == 0)["thresholds"] == {1: 0.23, 2: 0.71}
    for row in rows:
        assert row["thresholds"][2] - row["thresholds"][1] == pytest.approx(0.48)


def test_sweep_counts_negative_tiles_and_wrong_classes_as_fp_and_fn():
    box = [0, 0, 10, 10]
    other = [20, 0, 10, 10]
    ground_truth = {1: [(box, 1), (other, 2)], 2: []}
    predictions = {
        1: [(box, 1, 0.6), (other, 1, 0.7)],
        2: [(box, 1, 0.02)],
    }
    rows = sweep_confidence_thresholds(predictions, ground_truth, {1: 0.5, 2: 0.5}, step=0.5)
    assert [(row["TP"], row["FP"], row["FN"]) for row in rows] == [(1, 2, 1), (1, 1, 1), (0, 0, 2)]


@pytest.mark.parametrize("step", [0, -0.1, 1.1, math.nan, math.inf])
def test_sweep_rejects_invalid_steps(step):
    with pytest.raises(ValueError, match="Step"):
        sweep_confidence_thresholds({}, {}, {1: 0.5}, step=step)


@pytest.mark.parametrize("thresholds", [{}, {1: -0.1}, {1: 1.1}, {1: math.nan}])
def test_sweep_rejects_invalid_baselines(thresholds):
    with pytest.raises(ValueError, match="Baseline"):
        sweep_confidence_thresholds({}, {}, thresholds)


def test_sweep_handles_baseline_already_at_both_boundaries():
    rows = sweep_confidence_thresholds({}, {}, {1: 0, 2: 1})
    assert len(rows) == 1
    assert rows[0]["offset"] == 0


def test_sweep_rejects_unconfigured_classes_and_unknown_images():
    with pytest.raises(ValueError, match="class needs"):
        sweep_confidence_thresholds({}, {1: [([0, 0, 1, 1], 2)]}, {1: 0.5})
    with pytest.raises(ValueError, match="image IDs"):
        sweep_confidence_thresholds({2: []}, {1: []}, {1: 0.5})


@pytest.fixture
def report_dataset(tmp_path):
    categories = [
        {"id": 0, "name": "defects", "supercategory": "none"},
        {"id": 1, "name": "scratch"},
        {"id": 2, "name": "dent"},
    ]
    train_annotations = tmp_path / "training_categories.json"
    write_annotations(train_annotations, {"categories": categories})
    for split, names in (("valid", ["calibration.png", "negative.png"]), ("test", ["test.png"])):
        directory = tmp_path / split
        directory.mkdir()
        for name in names:
            with Image.new("RGB", (32, 32), "white") as image:
                image.save(directory / name)
        annotations = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}]
        if split == "valid":
            annotations.append({"id": 2, "image_id": 1, "category_id": 2, "bbox": [12, 0, 10, 10]})
        write_annotations(directory / "_annotations.coco.json", {
            "categories": categories,
            "images": [{"id": index, "file_name": name} for index, name in enumerate(names, 1)],
            "annotations": annotations,
        })

    def predict(path, threshold):
        assert threshold == 0.0
        if Path(path).name == "calibration.png":
            return SimpleNamespace(
                xyxy=[[0, 0, 10, 10], [12, 0, 22, 10]], class_id=[0, 1], confidence=[0.9, 0.6],
            )
        if Path(path).name == "negative.png":
            return SimpleNamespace(xyxy=[[0, 0, 10, 10]], class_id=[0], confidence=[0.2])
        return SimpleNamespace(
            xyxy=[[0, 0, 10, 10], [20, 20, 30, 30]], class_id=[0, 1], confidence=[0.3, 0.02],
        )

    model = SimpleNamespace(
        predict=Mock(side_effect=predict),
        model=SimpleNamespace(class_names=["scratch", "dent"]),
    )
    return model, train_annotations


def test_report_calibrates_on_valid_predicts_once_per_tile_and_exports(tmp_path, report_dataset):
    model, train_annotations = report_dataset
    report = build_report(model, tmp_path / "test", tmp_path / "valid", train_annotations, step=0.1)
    assert model.predict.call_count == 3
    assert [Path(call.args[0]).parent.name for call in model.predict.call_args_list] == ["valid", "valid", "test"]
    assert report["baseline_thresholds"] == {1: 0.25, 2: 0.25}
    assert report["evaluation_unit"] == "tile_detection"
    base = next(row for row in report["rows"] if row["offset"] == 0)
    assert (base["FP"], base["FN"]) == (0, 0)
    assert report["rows"][0]["FP"] == 1  # Includes the prediction below the usual 0.05 floor.
    assert report["rows"][-1]["FN"] == 1
    output = tmp_path / "report"
    save_report(report, output)
    with Image.open(output / "confidence_tradeoff.png") as chart:
        assert chart.width > 1000 and chart.height > 500
    assert "<svg" in (output / "confidence_tradeoff.svg").read_text(encoding="utf-8")
    saved = json.loads((output / "confidence_tradeoff.json").read_text(encoding="utf-8"))
    assert saved["rows"][0]["FP"] == 1
    with (output / "confidence_tradeoff.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(report["rows"])
    assert rows[0]["threshold_1"] == "0.0"
    assert rows[-1]["threshold_2"] == "1.0"


def test_binary_sweep_counts_original_images_not_defects_or_class_matches():
    box = [0, 0, 10, 10]
    ground_truth = {
        "detected": [(box, 1)],
        "wrong_class": [(box, 1)],
        "missed": [(box, 1)],
        "false_alarm": [],
        "quiet": [],
    }
    predictions = {
        "detected": [(box, 1, 0.9)],
        "wrong_class": [([20, 20, 10, 10], 2, 0.9)],
        "false_alarm": [(box, 1, 0.4)],
    }
    rows = sweep_confidence_thresholds(
        predictions, ground_truth, {1: 0.5, 2: 0.5}, step=0.5, binary=True,
    )
    assert [(row["TP"], row["FP"], row["FN"], row["TN"]) for row in rows] == [
        (2, 1, 1, 1), (2, 0, 1, 2), (0, 0, 3, 2),
    ]


def test_reconstructing_once_matches_filtering_tiles_before_nms_at_every_threshold():
    coco = {
        "source_images": [{"id": "door"}],
        "images": [
            {"id": 1, "file_name": "a.png", "source_image_id": "door", "tile_origin": [0, 0]},
            {"id": 2, "file_name": "b.png", "source_image_id": "door", "tile_origin": [16, 0]},
        ],
    }
    predictions = {
        1: [([20, 4, 8, 8], 1, 0.9)],
        2: [([4, 4, 8, 8], 1, 0.8), ([4, 4, 8, 8], 2, 0.7)],
    }
    ground_truth = {"door": [([20, 4, 8, 8], 1)]}
    rows = sweep_confidence_thresholds(
        reconstruct_predictions(coco, predictions), ground_truth, {1: 0.5, 2: 0.6}, step=0.1,
    )
    for row in rows:
        reconstructed = reconstruct_predictions(coco, filter_predictions(predictions, row["thresholds"]))
        metrics = detection_metrics(
            [(image_id, *pred) for image_id, preds in reconstructed.items() for pred in preds],
            ground_truth, iou_threshold=0.3,
        )
        assert all(row[key] == metrics[f"Total {key}"] for key in ("TP", "FP", "FN"))


def test_image_reports_align_originals_reconstruct_offsets_and_export_without_image_pixels(
    tmp_path, report_dataset,
):
    model, train_annotations = report_dataset
    test_dir = tmp_path / "test"
    coco = read_annotations(test_dir / "_annotations.coco.json")
    coco["images"] = [
        {"id": 1, "file_name": "test.png", "source_image_id": "door", "tile_origin": [0, 0]},
        {"id": 2, "file_name": "overlap.png", "source_image_id": "door", "tile_origin": [16, 0]},
        {"id": 3, "file_name": "negative_test.png", "source_image_id": "good", "tile_origin": [0, 0]},
    ]
    for filename in ("overlap.png", "negative_test.png"):
        with Image.new("RGB", (32, 32), "white") as image:
            image.save(test_dir / filename)
    coco["annotations"] = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [20, 4, 8, 8]},
        {"id": 2, "image_id": 2, "category_id": 1, "bbox": [4, 4, 8, 8]},
    ]
    coco["source_images"] = [
        {
            "id": "door", "file_name": str(Path("source_images") / "door.png"), "original_file_name": "door.tiff",
            "width": 64, "height": 32, "has_defects": False,
            "annotations": [{"id": 100, "category_id": 2, "bbox": [20, 4, 8, 8]}],
        },
        {
            "id": "good", "file_name": str(Path("source_images") / "good.png"), "original_file_name": "good.tiff",
            "width": 64, "height": 32, "has_defects": False, "annotations": [],
        },
    ]
    write_annotations(test_dir / "_annotations.coco.json", coco)
    originals = tmp_path / "originals"
    write_annotations(originals / "_annotations.coco.json", {
        "categories": [{"id": 7, "name": "scratch"}, {"id": 8, "name": "dent"}],
        "images": [
            {"id": 1, "file_name": "door.tiff", "width": 192, "height": 96},
            {"id": 2, "file_name": "good.tiff", "width": 192, "height": 96},
        ],
        "annotations": [{"id": 100, "image_id": 1, "category_id": 7, "bbox": [60, 12, 24, 24]}],
    })
    original_predict = model.predict.side_effect

    def predict(path, threshold):
        if Path(path).parent.name == "valid":
            return original_predict(path, threshold)
        assert threshold == 0.0
        filename = Path(path).name
        box = [20, 4, 28, 12] if filename == "test.png" else [4, 4, 12, 12]
        return SimpleNamespace(
            xyxy=[] if filename == "negative_test.png" else [box],
            class_id=[] if filename == "negative_test.png" else [0],
            confidence=[] if filename == "negative_test.png" else [0.9 if filename == "test.png" else 0.8],
        )

    model.predict.side_effect = predict
    before = (test_dir / "_annotations.coco.json").read_bytes()
    reports = build_reports(
        model, test_dir, tmp_path / "valid", train_annotations, step=0.1,
        original_data_dirs=[originals],
    )
    assert model.predict.call_count == 5
    assert set(reports) == {"tiles", "original_image_detection", "original_image_binary"}
    bases = {level: next(row for row in report["rows"] if row["offset"] == 0)
             for level, report in reports.items()}
    assert (bases["tiles"]["TP"], bases["tiles"]["FP"], bases["tiles"]["FN"]) == (2, 0, 0)
    assert (bases["original_image_detection"]["TP"], bases["original_image_detection"]["FP"]) == (1, 0)
    assert bases["original_image_binary"]["TP"] == bases["original_image_binary"]["TN"] == 1
    for level in ("original_image_detection", "original_image_binary"):
        report = reports[level]
        assert report["baseline_thresholds"] == reports["tiles"]["baseline_thresholds"]
        assert report["original_image_count"] == 2
        assert report["ground_truth_defects"] == report["defective_image_count"] == 1
        assert report["rows"][-1]["FN"] == 1
        save_report(report, tmp_path / level)
    with (tmp_path / "original_image_binary" / "confidence_tradeoff.csv").open(
        encoding="utf-8-sig", newline="",
    ) as f:
        saved = list(csv.DictReader(f))
    assert next(row for row in saved if float(row["offset"]) == 0)["TN"] == "1"
    assert (test_dir / "_annotations.coco.json").read_bytes() == before


def test_all_levels_skip_missing_metadata_but_explicit_images_fail_early(tmp_path, report_dataset, capsys):
    model, train_annotations = report_dataset
    with pytest.raises(ValueError, match="Original-image curves unavailable"):
        build_reports(model, tmp_path / "test", tmp_path / "valid", train_annotations, level="images")
    model.predict.assert_not_called()
    reports = build_reports(model, tmp_path / "test", tmp_path / "valid", train_annotations)
    assert set(reports) == {"tiles"}
    assert "source_images" in reports["tiles"]["original_image_skip_reason"]
    assert "WARNING: Original-image curves skipped" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "2", "invalid"])
def test_cli_rejects_invalid_step_before_loading_model(value):
    with pytest.raises(SystemExit) as error:
        main(["--step", value])
    assert error.value.code == 2
