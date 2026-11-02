import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from src.fuse_datasets import DatasetFuser
from src.preprocess.annotations import read_annotations, write_annotations
from src.test.__main__ import _align_categories, evaluate_dataset, main
from src.test.image_evaluation import (
    door_bucket,
    evaluate_images,
    reconstruct_predictions,
)
from src.test.metrics import binary_metrics, compute_map, detection_metrics
from src.test.matching import match_predictions
from src.test.thresholds import find_optimal_thresholds, select_thresholds


class ImageEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_annotations(self.root / "train" / "_annotations.coco.json", {
            "categories": [{"id": 0, "name": "defect"}, {"id": 1, "name": "other"}],
        })

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
        valid_dir = self.root / "valid"
        self.make_dataset(valid_dir)
        predictions = {
            "door0_tile0.png": [([4, 4, 12, 12], 0, 0.9)],
            "door1_tile1.png": [([4, 4, 12, 12], 0, 0.9)],
            "door2_tile0.png": [([4, 4, 12, 12], 0, 0.1)],
        }

        def predict(path, threshold):
            self.assertEqual(threshold, 0.05)
            if Path(path).parent == valid_dir:
                preds = {
                    "door0_tile0.png": [([4, 4, 12, 12], 0, 0.9)],
                    "door1_tile1.png": [([4, 4, 12, 12], 0, 0.4)],
                    "door2_tile0.png": [([4, 4, 12, 12], 0, 0.9)],
                }.get(Path(path).name, [])
            else:
                preds = predictions.get(Path(path).name, [])
            return SimpleNamespace(
                xyxy=np.array([pred[0] for pred in preds], dtype=np.float32),
                class_id=np.array([pred[1] for pred in preds]),
                confidence=np.array([pred[2] for pred in preds], dtype=np.float32),
            )

        model = SimpleNamespace(predict=Mock(side_effect=predict))
        with redirect_stdout(io.StringIO()):
            result = evaluate_dataset(model, test_dir, output_dir)
        calibration = result["threshold_calibration"]
        self.assertEqual(calibration["split"], "valid")
        self.assertEqual(calibration["selected_strategy"], "global")
        self.assertEqual(calibration["global_f1"], 1.0)
        self.assertEqual(calibration["global_threshold"], 0.45)
        self.assertEqual(
            [Path(call.args[0]).parent for call in model.predict.call_args_list],
            [valid_dir] * 8 + [test_dir] * 8,
        )
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

    def test_test_pipeline_applies_selected_per_class_thresholds(self):
        test_dir, output_dir = self.root / "test", self.root / "output"
        for directory in (test_dir, self.root / "valid"):
            coco = self.make_dataset(directory)
            coco["annotations"][1]["category_id"] = 1
            coco["source_images"][2]["annotations"][0]["category_id"] = 1
            write_annotations(directory / "_annotations.coco.json", coco)

        def predict(path, threshold):
            preds = {
                "door0_tile0.png": [([4, 4, 12, 12], 0, 0.2)],
                "door2_tile0.png": [([4, 4, 12, 12], 1, 0.9)],
                "door1_tile0.png": [([4, 4, 12, 12], 1, 0.2)],
            }.get(Path(path).name, [])
            return SimpleNamespace(
                xyxy=np.array([pred[0] for pred in preds]),
                class_id=np.array([pred[1] for pred in preds]),
                confidence=np.array([pred[2] for pred in preds]),
            )

        with redirect_stdout(io.StringIO()):
            result = evaluate_dataset(SimpleNamespace(predict=predict), test_dir, output_dir)
        calibration = result["threshold_calibration"]
        self.assertEqual(calibration["selected_strategy"], "per_class")
        self.assertEqual(calibration["per_class_f1"], 1.0)
        self.assertGreater(calibration["per_class_f1"], calibration["global_f1"])
        self.assertEqual(result["per_class_thresholds"], {"defect": 0.05, "other": 0.25})
        self.assertEqual(result["F1-Score"], 1.0)
        self.assertEqual(result["image_metrics"]["Total TN"], 2)

    def test_manual_global_threshold_skips_validation_and_filters_all_classes(self):
        test_dir = self.root / "test"
        coco = self.make_dataset(test_dir)
        coco["annotations"][1]["category_id"] = 1
        coco["source_images"][2]["annotations"][0]["category_id"] = 1
        write_annotations(test_dir / "_annotations.coco.json", coco)
        for threshold in (0.0, 0.02, 0.7, 1.0):
            with self.subTest(threshold=threshold):
                def predict(path, threshold):
                    self.assertEqual(threshold, 0.0)
                    self.assertEqual(Path(path).parent, test_dir)
                    return SimpleNamespace(
                        xyxy=np.array([[4, 4, 12, 12]]),
                        class_id=np.array([
                            1 if Path(path).name == "door2_tile0.png" else 0
                        ]),
                        confidence=np.array([0.02]),
                    )

                with patch("src.test.__main__.select_thresholds") as select:
                    with redirect_stdout(io.StringIO()):
                        result = evaluate_dataset(
                            SimpleNamespace(predict=predict), test_dir,
                            self.root / f"output_{threshold}", global_threshold=threshold,
                        )
                select.assert_not_called()
                self.assertEqual(
                    result["per_class_thresholds"], {"defect": threshold, "other": threshold}
                )
                self.assertEqual(result["threshold_calibration"]["source"], "manual")
                self.assertIsNone(result["threshold_calibration"]["split"])
                self.assertEqual(result["Total TP"], 2 if threshold <= 0.02 else 0)
                self.assertEqual(
                    result["image_metrics"]["Total TP"], 2 if threshold <= 0.02 else 0
                )

    def test_global_threshold_cli_forwards_value_and_preserves_default(self):
        model_class = Mock()
        rfdetr = SimpleNamespace(**{
            name: model_class for name in
            ("RFDETRLarge", "RFDETRMedium", "RFDETRSmall", "RFDETRNano")
        })
        config = SimpleNamespace(model_type="large", final_data=self.root)
        with patch.dict("sys.modules", {"rfdetr": rfdetr}):
            with patch("src.test.__main__.get_global_config", return_value=config):
                with patch("src.test.__main__.evaluate_dataset") as evaluate:
                    for argv, expected in (([], None), (["--global-threshold", "0.35"], 0.35)):
                        self.assertEqual(main(argv), 0)
                        self.assertEqual(evaluate.call_args.kwargs["global_threshold"], expected)
                    weights = self.root / "weights.pth"
                    snapshot = self.root / "training_categories.json"
                    write_annotations(snapshot, {"categories": []})
                    main(["--weights", str(weights)])
                    self.assertEqual(evaluate.call_args.kwargs["train_annotations"], snapshot)
                    original = self.root / "original.json"
                    main([
                        "--weights", str(weights), "--train-annotations", str(original),
                    ])
                    self.assertEqual(evaluate.call_args.kwargs["train_annotations"], original)

    def test_global_threshold_rejects_invalid_values_before_loading_model(self):
        for value in ("-0.1", "1.1", "nan", "inf", "invalid"):
            with self.subTest(value=value):
                with redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main(["--global-threshold", value])
                self.assertEqual(error.exception.code, 2)
        for threshold in (-0.1, 1.1, float("nan")):
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                evaluate_dataset(
                    Mock(), self.root / "test", self.root / "output",
                    global_threshold=threshold,
                )

    def test_calibration_rejects_missing_empty_or_shared_validation(self):
        test_dir, output_dir = self.root / "test", self.root / "output"
        self.make_dataset(test_dir)
        model = SimpleNamespace(predict=Mock())
        with self.assertRaises(FileNotFoundError):
            evaluate_dataset(model, test_dir, output_dir)
        with self.assertRaisesRegex(ValueError, "separate"):
            evaluate_dataset(model, test_dir, output_dir, valid_dir=test_dir)
        valid_dir = self.root / "valid"
        coco = self.make_dataset(valid_dir)
        coco["annotations"] = []
        write_annotations(valid_dir / "_annotations.coco.json", coco)
        with self.assertRaisesRegex(ValueError, "ground-truth"):
            evaluate_dataset(model, test_dir, output_dir)
        model.predict.assert_not_called()

    def test_alignment_remaps_tiles_and_sources_without_changing_input(self):
        coco = self.make_dataset(self.root / "test")
        coco["categories"] = [
            {"id": 0, "name": "untrained"}, {"id": 7, "name": "defect"},
        ]
        coco["annotations"][0]["category_id"] = 7
        coco["source_images"][0]["annotations"][0]["category_id"] = 7
        original = copy.deepcopy(coco)
        training = [{"id": 3, "name": "defect"}, {"id": 9, "name": "other"}]
        with redirect_stdout(io.StringIO()):
            aligned = _align_categories(coco, training)
        self.assertEqual(aligned["categories"], training)
        self.assertEqual([ann["category_id"] for ann in aligned["annotations"]], [3])
        self.assertEqual(
            aligned["source_images"][0]["annotations"][0]["category_id"], 3
        )
        self.assertTrue(aligned["source_images"][0]["has_defects"])
        self.assertEqual(aligned["source_images"][2]["annotations"], [])
        self.assertFalse(aligned["source_images"][2]["has_defects"])
        self.assertEqual(coco, original)

    def test_pipeline_aligns_each_split_by_name_to_training_ids(self):
        training = [{"id": 3, "name": "defect"}, {"id": 9, "name": "other"}]
        write_annotations(
            self.root / "train" / "_annotations.coco.json", {"categories": training}
        )
        for split, defect_id in (("valid", 7), ("test", 11)):
            directory = self.root / split
            coco = self.make_dataset(directory)
            coco["categories"] = [
                {"id": defect_id, "name": "defect"}, {"id": 3, "name": "untrained"},
            ]
            for ann in coco["annotations"]:
                ann["category_id"] = defect_id
            for source in coco["source_images"]:
                for ann in source["annotations"]:
                    ann["category_id"] = defect_id
            coco["annotations"].append({
                "id": 99, "image_id": 3, "category_id": 3, "bbox": [4, 4, 8, 8],
            })
            coco["source_images"][1]["annotations"] = [
                {"category_id": 3, "bbox": [4, 4, 8, 8]},
            ]
            coco["source_images"][1]["has_defects"] = True
            write_annotations(directory / "_annotations.coco.json", coco)

        def predict(path, threshold):
            positive = Path(path).name in ("door0_tile0.png", "door2_tile0.png")
            return SimpleNamespace(
                xyxy=np.array([[4, 4, 12, 12]]),
                class_id=np.array([3 if positive else 99]),
                confidence=np.array([0.9]),
            )

        with redirect_stdout(io.StringIO()):
            result = evaluate_dataset(
                SimpleNamespace(predict=predict), self.root / "test", self.root / "output"
            )
        self.assertEqual(result["threshold_calibration"]["global_f1"], 1.0)
        self.assertEqual(result["F1-Score"], 1.0)
        self.assertEqual(result["image_detection_metrics"]["F1-Score"], 1.0)
        self.assertEqual(result["image_metrics"]["Total TP"], 2)
        self.assertEqual(result["image_metrics"]["Total TN"], 2)
        self.assertEqual(
            read_annotations(self.root / "test" / "_annotations.coco.json")["categories"],
            [{"id": 11, "name": "defect"}, {"id": 3, "name": "untrained"}],
        )

    def test_map_counts_classes_with_no_predictions(self):
        ground_truth = {1: [([0, 0, 10, 10], 0), ([20, 20, 10, 10], 1)]}
        result = compute_map([(1, [0, 0, 10, 10], 0, 0.9)], ground_truth)
        self.assertAlmostEqual(result, 0.5)
        self.assertEqual(compute_map([], ground_truth), 0.0)

    def test_threshold_matching_orders_predictions_by_confidence(self):
        ground_truth = {1: [([0, 0, 10, 10], 0), ([8, 0, 10, 10], 0)]}
        predictions = [([3, 0, 10, 10], 0, 0.2), ([0, 0, 10, 10], 0, 0.9)]
        for ordered in (predictions, predictions[::-1]):
            with self.subTest(predictions=ordered):
                output = io.StringIO()
                with redirect_stdout(output):
                    thresholds = find_optimal_thresholds(
                        {1: ordered}, ground_truth, {0: "defect"},
                        iou_threshold=0.3, threshold_range=[0.1, 0.8],
                    )
                self.assertEqual(thresholds, {0: 0.1})
                self.assertIn("TP=2 FP=0 FN=0", output.getvalue())

    def test_threshold_search_warns_when_no_threshold_matches_ground_truth(self):
        output = io.StringIO()
        with redirect_stdout(output):
            find_optimal_thresholds(
                {1: [([0, 0, 10, 10], 1, 0.9)]},
                {1: [([0, 0, 10, 10], 0)]}, {0: "defect"},
                threshold_range=[0.1, 0.8],
            )
        self.assertIn("class_id=0  TP=0 FP=0 FN=1", output.getvalue())
        self.assertIn("not a calibrated optimum", output.getvalue())

    def test_global_threshold_can_beat_individually_optimized_class_f1(self):
        box = [0, 0, 10, 10]
        ground_truth = {
            1: [(box, 0), ([20, 0, 10, 10], 0)],
            2: [([20 * idx, 0, 10, 10], 1) for idx in range(10)],
        }
        predictions = {
            1: [
                (box, 0, 0.9), ([20, 0, 10, 10], 0, 0.2),
                *[([20 * idx, 0, 10, 10], 0, 0.2) for idx in range(2, 5)],
            ],
            2: [(box, 1, 0.9)],
        }
        with redirect_stdout(io.StringIO()):
            thresholds, report = select_thresholds(
                predictions, ground_truth, {0: "defect", 1: "other"},
                threshold_range=[0.1, 0.8],
            )
        self.assertEqual(report["per_class_thresholds"]["defect"], 0.8)
        self.assertGreater(report["global_f1"], report["per_class_f1"])
        self.assertEqual(report["selected_strategy"], "global")
        self.assertEqual(thresholds, {0: 0.1, 1: 0.1})
        self.assertAlmostEqual(report["global_f1"], 1 / 3)

    def test_strategy_comparison_counts_wrong_classes_and_negative_tiles(self):
        box = [0, 0, 10, 10]
        predictions = {
            1: [(box, 1, 0.9)],
            2: [(box, 0, 0.9), (box, 0, 0.8)],
            3: [(box, 1, 0.9)],
        }
        with redirect_stdout(io.StringIO()):
            thresholds, report = select_thresholds(
                predictions, {1: [(box, 0)], 2: [(box, 0)], 3: []},
                {0: "defect", 1: "other"}, threshold_range=[0.1],
            )
        # One TP, a duplicate FP, a wrong-class FP+FN, and a negative-tile FP.
        self.assertAlmostEqual(report["global_f1"], 1 / 3)
        self.assertEqual(report["per_class_f1"], report["global_f1"])
        self.assertEqual(report["selected_strategy"], "global")
        self.assertEqual(thresholds, {0: 0.1, 1: 0.1})

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

    def test_matching_prefers_valid_class_over_higher_iou_wrong_class(self):
        boxes = [[0, 0, 10, 10], [2, 0, 10, 10]]
        tp, fp, fn, mis = match_predictions(
            boxes, [0, 1], [0.9, 0.8], boxes[::-1], [0, 1], 0.5,
        )
        self.assertEqual(tp, {0, 1})
        self.assertEqual((fp, fn, mis), (set(), set(), set()))
        gt = {1: [(boxes[1], 0), (boxes[0], 1)]}
        preds = [(1, boxes[0], 0, 0.9), (1, boxes[1], 1, 0.8)]
        metrics = detection_metrics(preds, gt, 0.5)
        self.assertEqual(metrics["F1-Score"], 1.0)
        self.assertEqual(metrics["mAR@50"], 1.0)
        with redirect_stdout(io.StringIO()):
            _, report = select_thresholds(
                {1: [(boxes[0], 0, 0.9), (boxes[1], 1, 0.8)]},
                gt, {0: "a", 1: "b"}, iou_threshold=0.5, threshold_range=[0.1],
            )
        self.assertEqual(report["global_f1"], 1.0)

    def test_wrong_class_cannot_consume_gt_before_correct_lower_score_prediction(self):
        box = [0, 0, 10, 10]
        tp, fp, fn, mis = match_predictions(
            [box, box], [1, 0], [0.99, 0.8], [box], [0],
        )
        self.assertEqual(tp, {1})
        self.assertEqual(fp, {0})
        self.assertEqual(fn, set())
        self.assertEqual(mis, set())

    def test_checkpoint_label_order_mismatch_requires_original_training_mapping(self):
        test_dir = self.root / "test"
        self.make_dataset(test_dir)
        model = SimpleNamespace(
            model=SimpleNamespace(class_names=["other", "defect"]), predict=Mock(),
            # RF-DETR's public class_names property may use a different index base.
            class_names={1: "other", 2: "defect"},
        )
        with self.assertRaisesRegex(ValueError, "Checkpoint classes/order"):
            evaluate_dataset(model, test_dir, self.root / "output", global_threshold=0.5)
        model.predict.assert_not_called()
        original_training = self.root / "original_training.json"
        write_annotations(original_training, {
            "categories": [{"id": 0, "name": "other"}, {"id": 1, "name": "defect"}],
        })

        def predict(path, threshold):
            positive = Path(path).name in ("door0_tile0.png", "door2_tile0.png")
            return SimpleNamespace(
                xyxy=np.array([[4, 4, 12, 12]]) if positive else [],
                class_id=[1] if positive else [], confidence=[0.9] if positive else [],
            )

        model.predict.side_effect = predict
        with redirect_stdout(io.StringIO()):
            result = evaluate_dataset(
                model, test_dir, self.root / "output", global_threshold=0.5,
                train_annotations=original_training,
            )
        self.assertEqual(result["category_id_to_name"], {"0": "other", "1": "defect"})
        self.assertEqual(result["F1-Score"], 1.0)
        self.assertEqual(result["image_detection_metrics"]["F1-Score"], 1.0)

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
