import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from src.fuse_datasets import DatasetFuser
from src.preprocess.annotations import read_annotations, write_annotations
from src.test.__main__ import evaluate_dataset
from src.test.image_evaluation import (
    door_bucket,
    evaluate_images,
    reconstruct_predictions,
)
from src.test.metrics import binary_metrics, compute_map, detection_metrics


class ImageEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_dataset(self, directory):
        coco = {
            "images": [], "annotations": [], "source_images": [],
            "categories": [{"id": 0, "name": "defect"}, {"id": 1, "name": "other"}],
        }
        for idx in range(4):
            source_id = f"door{idx}"
            filename = str(Path("source_images") / f"{source_id}.png")
            full_path = directory / filename
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.new("RGB", (64, 32), "white") as image:
                image.save(full_path)
            anns = (
                [{"bbox": [4, 4, 8, 8], "category_id": 0}] if idx in (0, 2) else []
            )
            coco["source_images"].append({
                "id": source_id,
                "file_name": filename,
                "original_file_name": f"{source_id}.tiff",
                "width": 64, "height": 32,
                "has_defects": bool(anns),
                "annotations": anns,
            })
            for tile_idx in range(2):
                image_id = idx * 2 + tile_idx + 1
                tile_name = f"{source_id}_tile{tile_idx}.png"
                with Image.new("RGB", (32, 32), "white") as tile:
                    tile.save(directory / tile_name)
                coco["images"].append({
                    "id": image_id, "file_name": tile_name,
                    "width": 32, "height": 32, "source_image_id": source_id,
                    "tile_origin": [tile_idx * 32, 0],
                })
                if tile_idx == 0:
                    coco["annotations"].extend(
                        {**ann, "id": image_id, "image_id": image_id} for ann in anns
                    )
        write_annotations(directory / "_annotations.coco.json", coco)
        return coco

    def test_door_buckets_depend_only_on_presence(self):
        for has_defects, has_predictions, expected in (
            (True, True, "TP"), (False, True, "FP"),
            (True, False, "FN"), (False, False, "TN"),
        ):
            self.assertEqual(door_bucket(has_defects, has_predictions), expected)

    def test_reconstructs_offsets_and_suppresses_overlap_per_class(self):
        coco = {
            "source_images": [{"id": "door"}],
            "images": [
                {"id": 1, "file_name": "a.png", "source_image_id": "door", "tile_origin": [0, 10]},
                {"id": 2, "file_name": "b.png", "source_image_id": "door", "tile_origin": [16, 10]},
            ],
        }
        predictions = {
            1: [([20, 4, 8, 8], 0, 0.9)],
            2: [([4, 4, 8, 8], 0, 0.8), ([4, 4, 8, 8], 1, 0.7)],
        }
        original = copy.deepcopy(predictions)
        result = reconstruct_predictions(coco, predictions)
        self.assertEqual(result["door"], [
            ([20, 14, 8, 8], 0, 0.9), ([20, 14, 8, 8], 1, 0.7),
        ])
        self.assertEqual(predictions, original)

    def test_complete_images_rendered_once_into_all_four_buckets(self):
        test_dir, output_dir = self.root / "test", self.root / "doors"
        coco = self.make_dataset(test_dir)
        # A wrong-class detection far from GT still makes this door a binary TP.
        predictions = {
            2: [([20, 20, 4, 4], 1, 0.9)],
            4: [([4, 4, 8, 8], 0, 0.9)],
        }
        with redirect_stdout(io.StringIO()):
            binary, detection = evaluate_images(coco, predictions, test_dir, output_dir)
        for bucket, filename in (
            ("TP", "door0.png"), ("FP", "door1.png"),
            ("FN", "door2.png"), ("TN", "door3.png"),
        ):
            paths = list((output_dir / bucket).glob("*.png"))
            self.assertEqual([path.name for path in paths], [filename])
            self.assertEqual(cv2.imread(str(paths[0])).shape, (32, 128, 3))
        for key in ("Precision", "Recall", "F1-Score", "Accuracy", "Specificity"):
            self.assertEqual(binary[key], 0.5)
        for bucket in ("TP", "FP", "FN", "TN"):
            self.assertEqual(binary[f"Total {bucket}"], 1)
        self.assertEqual(detection["mAP@50"], 0.0)
        self.assertEqual(detection["mAR@50"], 0.0)

    def test_test_pipeline_uses_filtered_predictions_for_door_decisions(self):
        test_dir, output_dir = self.root / "test", self.root / "output"
        self.make_dataset(test_dir)
        predictions = {
            "door0_tile0.png": [([4, 4, 12, 12], 0, 0.9)],
            "door1_tile1.png": [([4, 4, 12, 12], 0, 0.9)],
            "door2_tile0.png": [([4, 4, 12, 12], 0, 0.1)],
        }

        def predict(path, threshold):
            self.assertEqual(threshold, 0.05)
            preds = predictions.get(Path(path).name, [])
            return SimpleNamespace(
                xyxy=np.array([pred[0] for pred in preds], dtype=np.float32),
                class_id=np.array([pred[1] for pred in preds]),
                confidence=np.array([pred[2] for pred in preds], dtype=np.float32),
            )

        with patch("src.test.__main__.find_optimal_thresholds", return_value={0: 0.5}):
            with redirect_stdout(io.StringIO()):
                result = evaluate_dataset(SimpleNamespace(predict=predict), test_dir, output_dir)
        self.assertEqual(result["Total TP"], 1)
        self.assertEqual(result["Total FP"], 1)
        self.assertEqual(result["Total FN"], 1)
        self.assertEqual(result["image_metrics"]["Total TN"], 1)
        self.assertEqual(result["image_metrics"]["Total FN"], 1)
        self.assertEqual(result["image_detection_metrics"]["mAR@50"], 0.5)
        self.assertGreater(result["image_detection_metrics"]["mAP@50"], 0.0)
        self.assertEqual(read_annotations(output_dir / "metrics.json"), result)
        self.assertTrue((output_dir / "TP" / "door0_tile0.png").is_file())
        self.assertTrue((output_dir / "FN" / "door2_tile0.png").is_file())
        self.assertTrue((output_dir / "doors" / "TN" / "door3.png").is_file())

    def test_map_counts_classes_with_no_predictions(self):
        ground_truth = {1: [([0, 0, 10, 10], 0), ([20, 20, 10, 10], 1)]}
        result = compute_map([(1, [0, 0, 10, 10], 0, 0.9)], ground_truth)
        self.assertAlmostEqual(result, 0.5)
        self.assertEqual(compute_map([], ground_truth), 0.0)

    def test_full_image_detection_metrics_match_original_boxes_not_duplicate_tiles(self):
        test_dir = self.root / "test"
        coco = self.make_dataset(test_dir)
        coco["source_images"] = coco["source_images"][:1]
        coco["source_images"][0]["annotations"] = [
            {"bbox": [20, 4, 8, 8], "category_id": 0}
        ]
        coco["images"] = coco["images"][:2]
        coco["images"][1]["tile_origin"] = [16, 0]
        predictions = {
            1: [([20, 4, 8, 8], 0, 0.9)],
            2: [([4, 4, 8, 8], 0, 0.8)],
        }
        with redirect_stdout(io.StringIO()):
            binary, metrics = evaluate_images(coco, predictions, test_dir, self.root / "doors")
        self.assertEqual(binary["Total TP"], 1)
        self.assertEqual(metrics["Total TP"], 1)
        self.assertEqual(metrics["Total FP"], 0)
        for key in ("mAP@50", "mAP@50:95", "mAR@50", "mAR@50:95"):
            self.assertEqual(metrics[key], 1.0)

    def test_detection_metrics_count_wrong_class_as_fp_and_fn(self):
        metrics = detection_metrics(
            [(1, [0, 0, 10, 10], 0, 0.9), (1, [20, 20, 10, 10], 0, 0.8)],
            {1: [([0, 0, 10, 10], 0), ([20, 20, 10, 10], 1)]},
        )
        self.assertEqual(metrics["Total TP"], 1)
        self.assertEqual(metrics["Total FP"], 1)
        self.assertEqual(metrics["Total FN"], 1)
        self.assertEqual(metrics["Recall"], 0.5)

    def test_binary_metrics_handle_no_positive_doors(self):
        metrics = binary_metrics({"TP": 0, "FP": 0, "FN": 0, "TN": 4})
        self.assertEqual(metrics["Accuracy"], 1.0)
        self.assertEqual(metrics["Specificity"], 1.0)
        self.assertEqual(metrics["F1-Score"], 0.0)

    def test_legacy_or_incomplete_tile_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "preprocesado"):
            reconstruct_predictions({"images": []}, {})
        with self.assertRaisesRegex(ValueError, "origen"):
            reconstruct_predictions({
                "source_images": [{"id": "door"}],
                "images": [{"id": 1, "file_name": "tile.png"}],
            }, {})

    def test_fusion_preserves_complete_images_origins_and_category_mapping(self):
        paths = [self.root / name for name in ("master", "slave")]
        for idx, path in enumerate(paths):
            coco = self.make_dataset(path)
            source = coco["source_images"][0]
            old_path = path / source["file_name"]
            source["id"] = f"source{idx}"
            source["file_name"] = str(Path("source_images") / f"source{idx}.png")
            old_path.rename(path / source["file_name"])
            tile = coco["images"][0]
            old_tile_path = path / tile["file_name"]
            tile.update(file_name=f"tile{idx}.png", source_image_id=source["id"])
            old_tile_path.rename(path / tile["file_name"])
            coco.update(
                source_images=[source], images=[tile],
                annotations=[coco["annotations"][0]],
                categories=[{"id": 0, "name": f"class{idx}"}],
            )
            write_annotations(path / "_annotations.coco.json", coco)
        output = self.root / "fused"
        DatasetFuser.fuse_datasets(*paths, output)
        result = read_annotations(output / "_annotations.coco.json")
        self.assertEqual(len(result["source_images"]), 2)
        self.assertEqual(len({img["id"] for img in result["images"]}), 2)
        for idx, source in enumerate(result["source_images"]):
            self.assertTrue((output / source["file_name"]).is_file())
            self.assertEqual(source["annotations"][0]["category_id"], idx)
            self.assertEqual(result["images"][idx]["source_image_id"], source["id"])
            self.assertEqual(result["images"][idx]["tile_origin"], [0, 0])
        self.assertEqual(result["annotations"][1]["category_id"], 1)
        with redirect_stdout(io.StringIO()):
            binary, _ = evaluate_images(result, {}, output, self.root / "fused_doors")
        self.assertEqual(binary["Total FN"], 2)


if __name__ == "__main__":
    unittest.main()
