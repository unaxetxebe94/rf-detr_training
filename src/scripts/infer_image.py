"""Run tiled RF-DETR inference and reconstruct detections on an original image."""

import argparse
import io
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.scripts"

from ..core.params import get_data_config, get_global_config, get_preprocess_config
from ..preprocess.annotations import read_annotations, write_annotations
from ..preprocess.image import open_image
from ..preprocess.tile_creation import TILE_SIZE_MAPPER, iter_tile_regions, prepare_image_for_tiles
from ..test.categories import validate_checkpoint_categories
from ..test.image_evaluation import suppress_overlapping_predictions


def infer_image(model, image_path, *, tile_size, cfg, threshold=0.5, nms_iou=0.5, categories=None):
    if not 0 <= threshold <= 1 or not 0 <= nms_iou <= 1:
        raise ValueError("Confidence and NMS IoU must be between 0 and 1.")
    if tile_size <= 0 or not math.isfinite(cfg.resize) or cfg.resize <= 0:
        raise ValueError("Tile size and resize must be positive and finite.")
    image_path = Path(image_path)
    original = open_image(str(image_path))
    img, roi, scale = prepare_image_for_tiles(original, image_path, tile_size, cfg)
    offset_x, offset_y = (roi.x, roi.y) if roi else (0, 0)
    allowed_labels = None if categories is None else {cat["id"] for cat in categories}
    predictions = []
    tile_count = 0
    for x, y, width, height in iter_tile_regions(img, tile_size):
        tile = img.crop(x, y, width, height)
        # A PNG buffer preserves exactly the pixels saved during preprocessing.
        with Image.open(io.BytesIO(tile.write_to_buffer(".png"))) as tile_image:
            prediction = model.predict(tile_image.convert("RGB"), threshold=threshold)
        for box, label, score in zip(
            prediction.xyxy, prediction.class_id, prediction.confidence, strict=True,
        ):
            label, score = int(label) + 1, float(score)
            if score < threshold or (allowed_labels is not None and label not in allowed_labels):
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            x1, x2 = (min(width, max(0, value)) for value in (x1, x2))
            y1, y2 = (min(height, max(0, value)) for value in (y1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            x1, x2 = (min(original.width, (x + value) / scale + offset_x) for value in (x1, x2))
            y1, y2 = (min(original.height, (y + value) / scale + offset_y) for value in (y1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            predictions.append((
                [x1, y1, x2 - x1, y2 - y1],
                label, score,
            ))
        tile_count += 1
    predictions = suppress_overlapping_predictions(predictions, nms_iou)
    return {
        "image": str(image_path.resolve()),
        "width": original.width,
        "height": original.height,
        "coordinate_space": "original_image_pixels",
        "bbox_format": "xywh",
        "roi": None if roi is None else [roi.x, roi.y, roi.w, roi.h],
        "resize_scale": scale,
        "tile_size": tile_size,
        "tile_count": tile_count,
        "threshold": threshold,
        "nms_iou": nms_iou,
        "categories": categories or [],
        "detections": [
            {"bbox": box, "category_id": label, "score": score}
            for box, label, score in predictions
        ],
    }


def save_preview(image_path, result, destination, max_size=2048):
    if max_size <= 0:
        raise ValueError("Preview size must be positive.")
    original = open_image(str(image_path))
    thumb = original.thumbnail_image(max_size, height=max_size, size="down")
    with Image.open(io.BytesIO(thumb.write_to_buffer(".png"))) as image:
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    scale_x, scale_y = image.width / result["width"], image.height / result["height"]
    labels = {cat["id"]: cat["name"] for cat in result["categories"]}
    for detection in result["detections"]:
        x, y, width, height = detection["bbox"]
        box = (x * scale_x, y * scale_y, (x + width) * scale_x, (y + height) * scale_y)
        draw.rectangle(box, outline="red", width=2)
        label = labels.get(detection["category_id"], str(detection["category_id"]))
        draw.text((box[0], max(0, box[1] - 12)), f"{label} {detection['score']:.2f}", fill="red")
    image.save(destination)


def _probability(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Value must be between 0 and 1.")
    return number


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inferencia de una imagen grande por tiles, sin anotaciones de entrada.",
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, help="Defaults to the latest dated training checkpoint.")
    parser.add_argument("--model-type", choices=tuple(TILE_SIZE_MAPPER), help="Defaults to params.yaml.")
    parser.add_argument("--train-annotations", type=Path, help="Training COCO or training_categories.json.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to inference_output/<image stem>.")
    parser.add_argument("--threshold", type=_probability, default=0.5)
    parser.add_argument("--nms-iou", type=_probability, default=0.5)
    parser.add_argument("--preview-size", type=_positive_int, default=2048)
    args = parser.parse_args(argv)
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    weights = args.weights
    if weights is None:
        from ..test.run_test import get_latest_model_weights

        weights = Path(get_latest_model_weights())
    if not weights.is_file():
        raise FileNotFoundError(weights)
    train_annotations = args.train_annotations
    if train_annotations is None:
        train_annotations = weights.parent / "training_categories.json"
        if not train_annotations.is_file():
            train_annotations = get_data_config().training_dataset_path / "train" / "_annotations.coco.json"
    categories = [
        cat for cat in read_annotations(train_annotations)["categories"]
        if cat.get("supercategory") != "none"
    ]
    if not categories or sorted(cat["id"] for cat in categories) != list(range(1, len(categories) + 1)):
        raise ValueError("Training categories must have contiguous IDs starting at 1.")
    model_type = args.model_type or get_global_config().model_type
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    models = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}
    model = models[model_type](pretrain_weights=str(weights.resolve()))
    validate_checkpoint_categories(model, categories)
    result = infer_image(
        model, args.image, tile_size=TILE_SIZE_MAPPER[model_type], cfg=get_preprocess_config(),
        threshold=args.threshold, nms_iou=args.nms_iou, categories=categories,
    )
    result["weights"] = str(weights.resolve())
    result["model_type"] = model_type
    output_dir = args.output_dir or Path("inference_output") / args.image.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    write_annotations(output_dir / "predictions.json", result)
    save_preview(args.image, result, output_dir / "preview.png", args.preview_size)
    print(f"{result['tile_count']} tiles; {len(result['detections'])} detections. Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
