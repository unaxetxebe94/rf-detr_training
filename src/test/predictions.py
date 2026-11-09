"""Model inference and confidence filtering in the evaluation label space."""

from .iou import xyxy_to_xywh

RAW_THRESHOLD = 0.05
DEFAULT_THRESHOLD = 0.5


def predict_dataset(model, directory, coco, raw_threshold=RAW_THRESHOLD):
    allowed_labels = {cat["id"] for cat in coco["categories"]}
    raw_predictions = {}
    for image in coco["images"]:
        image_path = directory / image["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        prediction = model.predict(str(image_path), threshold=raw_threshold)
        raw_predictions[image["id"]] = [
            (xyxy_to_xywh(box), int(label) + 1, float(score))
            for box, label, score in zip(
                prediction.xyxy, prediction.class_id, prediction.confidence, strict=True
            )
            if int(label) + 1 in allowed_labels
        ]
        print(f"{image['file_name']}: {len(raw_predictions[image['id']])} detecciones.")
    return raw_predictions


def filter_predictions(predictions_by_image, thresholds, default_threshold=DEFAULT_THRESHOLD):
    return {
        image_id: [
            (box, label, score) for box, label, score in predictions
            if score >= thresholds.get(label, default_threshold)
        ]
        for image_id, predictions in predictions_by_image.items()
    }
