import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from src.preprocess import tile_creation
from src.preprocess.roi import ROI
from src.scripts import infer_image


def prediction(boxes=(), labels=(), scores=()):
    return SimpleNamespace(xyxy=boxes, class_id=labels, confidence=scores)


@pytest.mark.parametrize(
    "size,resize,roi",
    [
        ((19, 13), 0.5, None), ((3, 5), 0.5, None),
        ((31, 23), 0.5, ROI(4, 3, 23, 17)), ((31, 23), 0.5, ROI(4, 3, 3, 5)),
    ],
)
def test_inference_uses_exact_preprocessing_pixels_and_origins(tmp_path, monkeypatch, size, resize, roi):
    pixels = np.random.default_rng(42).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    image_path = tmp_path / "input.png"
    Image.fromarray(pixels).save(image_path)
    cfg = SimpleNamespace(
        resize=resize, apply_roi=roi is not None, min_retained_dimension_ratio=0.1,
        saving_prob=1.0, n_jobs=1,
    )
    monkeypatch.setattr(tile_creation, "get_global_config", lambda: SimpleNamespace(model_type="nano", seed=42))
    monkeypatch.setattr(tile_creation, "get_preprocess_config", lambda: cfg)
    monkeypatch.setitem(tile_creation.TILE_SIZE_MAPPER, "nano", 8)
    monkeypatch.setattr(tile_creation, "detect_roi", lambda *args: roi)
    coco = {
        "images": [{"id": 1, "file_name": image_path.name}],
        "annotations": [], "categories": [],
    }
    tiles_dir = tmp_path / "tiles"
    tile_creation.create_tiles(tiles_dir, input_dir=tmp_path, annotations=coco)
    tiles = json.loads((tiles_dir / "_annotations.coco.json").read_text())["images"]
    seen = []

    def predict(image, threshold):
        seen.append(np.asarray(image).copy())
        return prediction([[0, 0, 1, 1]], [0], [0.9])

    result = infer_image.infer_image(
        SimpleNamespace(predict=predict), image_path, tile_size=8, cfg=cfg,
    )
    assert result["tile_count"] == len(tiles) == len(seen)
    assert result["width"] == size[0]
    assert result["height"] == size[1]
    crop_size = (roi.w, roi.h) if roi else size
    expected_scale = max(8 / crop_size[0], 8 / crop_size[1]) if min(crop_size) < 8 else resize
    assert result["resize_scale"] == expected_scale
    for tile, actual, detection in zip(tiles, seen, result["detections"], strict=True):
        with Image.open(tiles_dir / tile["file_name"]) as expected:
            np.testing.assert_array_equal(actual, np.asarray(expected))
        x, y = tile["tile_origin"]
        assert detection["bbox"] == pytest.approx(
            [x / expected_scale, y / expected_scale, 1 / expected_scale, 1 / expected_scale],
        )


def test_overlapping_tiles_are_merged_per_class_and_clipped(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (12, 8)).save(image_path)
    model = Mock()
    model.predict.side_effect = [
        prediction([[5, 2, 7, 4], [-5, -2, 0, 0]], [0, 0], [0.9, 0.9]),
        prediction([[1, 2, 3, 4]] * 4, [0, 1, 0, 10], [0.8, 0.7, 0.1, 0.99]),
    ]
    result = infer_image.infer_image(
        model, image_path, tile_size=8, cfg=SimpleNamespace(resize=1.0, apply_roi=False),
    )
    assert result["tile_count"] == 2
    assert result["detections"] == [
        {"bbox": [5, 2, 2, 2], "category_id": 11, "score": 0.99},
        {"bbox": [5, 2, 2, 2], "category_id": 1, "score": 0.9},
        {"bbox": [5, 2, 2, 2], "category_id": 2, "score": 0.7},
    ]


@pytest.mark.parametrize("with_annotations", [False, True])
def test_cli_saves_original_coordinates_and_preview_without_reading_annotations(
    tmp_path, monkeypatch, with_annotations,
):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (24, 16)).save(image_path)
    weights = tmp_path / "checkpoint.pth"
    weights.touch()
    if with_annotations:
        (tmp_path / "training_categories.json").write_text("Not a valid JSON catalog")
        (tmp_path / "_annotations.coco.json").write_text("Not valid COCO annotations")
    model = SimpleNamespace(predict=Mock(return_value=prediction([[1, 2, 5, 6]], [0], [0.9])))
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(
        RFDETRNano=factory, RFDETRSmall=factory, RFDETRMedium=factory, RFDETRLarge=factory,
    ))
    monkeypatch.setitem(tile_creation.TILE_SIZE_MAPPER, "nano", 8)
    monkeypatch.setattr(infer_image, "get_preprocess_config", lambda: SimpleNamespace(resize=0.5, apply_roi=False))
    output = tmp_path / "output"
    assert infer_image.main([
        str(image_path), "--weights", str(weights), "--model-type", "nano",
        "--output-dir", str(output), "--preview-size", "12",
    ]) == 0
    result = json.loads((output / "predictions.json").read_text())
    assert result["coordinate_space"] == "original_image_pixels"
    assert result["detections"][0]["bbox"] == [2, 4, 8, 8]
    assert result["weights"] == str(weights.resolve())
    with Image.open(output / "preview.png") as preview:
        assert preview.size == (12, 8)
    factory.assert_called_once_with(pretrain_weights=str(weights))


def test_empty_predictions_still_save_preview(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8)).save(image_path)
    result = infer_image.infer_image(
        SimpleNamespace(predict=lambda *args, **kwargs: prediction()),
        image_path, tile_size=8, cfg=SimpleNamespace(resize=1.0, apply_roi=False),
    )
    assert result["detections"] == []
    infer_image.save_preview(image_path, result, tmp_path / "preview.png")
    assert (tmp_path / "preview.png").is_file()


def test_preview_draws_only_boxes(tmp_path, monkeypatch):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (20, 10)).save(image_path)
    draw = Mock()
    monkeypatch.setattr(infer_image.ImageDraw, "Draw", lambda image: draw)
    infer_image.save_preview(
        image_path,
        {"width": 20, "height": 10, "detections": [{"bbox": [2, 2, 8, 6]}]},
        tmp_path / "preview.png", max_size=10,
    )
    draw.rectangle.assert_called_once_with((1, 1, 5, 4), outline="red", width=2)
    draw.text.assert_not_called()


def test_resize_rounding_does_not_produce_boxes_outside_original(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (19, 13)).save(image_path)
    result = infer_image.infer_image(
        SimpleNamespace(predict=lambda *args, **kwargs: prediction([[-1, -1, 9, 9]], [0], [0.9])),
        image_path, tile_size=8, cfg=SimpleNamespace(resize=0.5, apply_roi=False), nms_iou=1.0,
    )
    assert result["detections"][-1]["bbox"] == [4, 0, 15, 13]


@pytest.mark.parametrize("resize", [0, -1, float("nan"), float("inf")])
def test_invalid_resize_rejected_before_opening_image(resize):
    with pytest.raises(ValueError, match="resize"):
        infer_image.infer_image(
            None, "missing.png", tile_size=8, cfg=SimpleNamespace(resize=resize, apply_roi=False),
        )
