import copy
import io
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from core.params import get_data_config, get_global_config, get_preprocess_config
from src.preprocess.__main__ import main
from src.preprocess.annotations import (
    SPLITS,
    correct_labels,
    read_annotations,
    write_annotations,
)
from src.preprocess.augmenter import augment_dataset
from src.preprocess.splitter import split_dataset


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = [self.root / "alpha", self.root / "beta"]
        self.trainings = self.root / "trainings"
        self.params = {
            "task-name": "test",
            "model-type": "NANO",
            "seed": 42,
            "data": {
                "sources": [str(source) for source in self.sources],
                "training-dataset-path": str(self.trainings),
            },
            "preprocess": {
                "saving-prob": 1.0,
                "min-retained-dimension-ratio": 0.10,
                "resize": 0.5,
                "apply-roi": False,
                "train-ratio": 0.7,
                "test-ratio": 0.15,
                "val-ratio": 0.15,
                "augmentations-per-image": 1,
                "max-transforms-per-sample": 5,
            },
        }
        # El proyecto mezcla los imports `core.params` (por PYTHONPATH=src) y
        # `src.core.params` (paquete desde la raíz); Python los carga como
        # módulos distintos, así que parcheamos read_params en ambos.
        config_patches = [
            patch("src.core.params.read_params", return_value=self.params),
            patch("core.params.read_params", return_value=self.params),
        ]
        for config_patch in config_patches:
            config_patch.start()
            self.addCleanup(config_patch.stop)

    def make_dataset(self, directory, count=20):
        directory.mkdir(parents=True, exist_ok=True)
        coco = {
            "info": {"description": "test"},
            "images": [],
            "annotations": [],
            "categories": [{"id": 7, "name": "a"}, {"id": 20, "name": "b"}],
        }
        for idx in range(count):
            name = f"image{idx}.png"
            with Image.new("RGB", (32, 24), (80 + idx, 120, 160)) as image:
                image.save(directory / name)
            coco["images"].append({
                "id": idx + 1, "file_name": name, "width": 32, "height": 24,
            })
            coco["annotations"].append({
                "id": idx + 1,
                "image_id": idx + 1,
                "category_id": 7 if idx % 2 else 20,
                "bbox": [2, 2, 20, 16],
                "area": 320,
                "segmentation": [[2, 2, 22, 2, 22, 18, 2, 18]],
                "iscrowd": 0,
            })
        write_annotations(directory / "_annotations.coco.json", coco)
        return coco

    def source_output(self, idx):
        """Directorio de salida que genera el pipeline para la fuente i-ésima."""
        name = Path(self.params["data"]["sources"][idx]).name
        return self.trainings / "sources" / f"{name}_source{idx}"

    def test_config_reads_sources_list_and_preprocess_defaults(self):
        data_cfg = get_data_config()
        self.assertEqual(data_cfg.sources, self.sources)
        self.assertEqual(data_cfg.training_dataset_path, self.trainings)
        self.assertEqual(get_global_config().task_name, "test")
        self.assertEqual(get_global_config().model_type, "nano")
        self.assertEqual(get_global_config().seed, 42)
        self.assertEqual(get_preprocess_config().min_retained_dimension_ratio, 0.1)

    def test_global_config_accepts_nested_global_and_underscore_keys(self):
        global_values = {
            key.replace("-", "_"): value
            for key, value in self.params.items()
            if key != "preprocess" and key != "data"
        }
        self.params.clear()
        self.params.update({"global": global_values})
        global_cfg = get_global_config()
        self.assertEqual(global_cfg.task_name, "test")
        self.assertEqual(global_cfg.model_type, "nano")
        self.assertEqual(global_cfg.seed, 42)

    def test_pipeline_processes_every_source_in_data_sources_order(self):
        for source in self.sources:
            self.make_dataset(source)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([]), 0)
        for idx, source in enumerate(self.sources):
            with self.subTest(source=source.name, idx=idx):
                # El pipeline escribe en sources/<nombre>_source<indice>
                self.assertIn(
                    f"{source.name}_source{idx}",
                    [path.name for path in (self.trainings / "sources").iterdir()],
                )
                output = self.source_output(idx)
                original_ids = []
                augmented_count = 0
                category_schemas = []
                for split in SPLITS:
                    coco = read_annotations(output / split / "_annotations.coco.json")
                    category_schemas.append(coco["categories"])
                    images = {img["id"]: img for img in coco["images"]}
                    self.assertEqual(len(images), len(coco["images"]))
                    self.assertEqual(
                        len({ann["id"] for ann in coco["annotations"]}),
                        len(coco["annotations"]),
                    )
                    category_ids = {cat["id"] for cat in coco["categories"]}
                    self.assertEqual(category_ids, set(range(len(category_ids))))
                    for img in images.values():
                        self.assertEqual((img["width"], img["height"]), (16, 12))
                        self.assertTrue((output / split / img["file_name"]).is_file())
                        if "_aug" in img["file_name"]:
                            self.assertEqual(split, "train")
                            augmented_count += 1
                        else:
                            original_ids.append(img["source_image_id"])
                    for ann in coco["annotations"]:
                        self.assertIn(ann["image_id"], images)
                        self.assertIn(ann["category_id"], category_ids)
                        self.assertEqual(ann["bbox"], [1, 1, 10, 8])
                self.assertEqual(len(original_ids), 20)
                self.assertEqual(len(set(original_ids)), 20)
                self.assertTrue(
                    all(cats == category_schemas[0] for cats in category_schemas)
                )
                train = read_annotations(output / "train" / "_annotations.coco.json")
                self.assertEqual(augmented_count * 2, len(train["images"]))
                self.assertGreater(augmented_count, 0)

                from src.test.image_evaluation import evaluate_images

                test = read_annotations(output / "test" / "_annotations.coco.json")
                predictions = {}
                for ann in test["annotations"]:
                    predictions.setdefault(ann["image_id"], []).append(
                        (ann["bbox"], ann["category_id"], 0.9)
                    )
                with redirect_stdout(io.StringIO()):
                    binary, detection = evaluate_images(
                        test, predictions, output / "test", output / "door_evaluation"
                    )
                self.assertEqual(binary["Total TP"], len(test["source_images"]))
                self.assertEqual(detection["mAP@50:95"], 1.0)

    def test_pipeline_splits_sources_before_sampling_training_tiles(self):
        source = self.sources[0]
        self.params["data"]["sources"] = [str(source)]
        original = self.make_dataset(source, count=30)
        original["annotations"] = [
            ann for ann in original["annotations"] if ann["image_id"] % 2
        ]
        for img in original["images"]:
            img.update(width=800, height=64)
            with Image.new("RGB", (800, 64), "white") as image:
                image.save(source / img["file_name"])
        write_annotations(source / "_annotations.coco.json", original)
        cfg = self.params["preprocess"]
        cfg.update({"saving-prob": 0.0, "resize": 1.0, "augmentations-per-image": 0})
        with redirect_stdout(io.StringIO()):
            main([])
        output = self.source_output(0)
        seen_sources = set()
        for split in SPLITS:
            coco = read_annotations(output / split / "_annotations.coco.json")
            source_ids = {img["source_image_id"] for img in coco["images"]}
            self.assertTrue(source_ids.isdisjoint(seen_sources))
            seen_sources.update(source_ids)
            if split == "train":
                annotated_ids = {ann["image_id"] for ann in coco["annotations"]}
                self.assertEqual(annotated_ids, {img["id"] for img in coco["images"]})
            else:
                self.assertEqual(len(coco["images"]), 3 * len(source_ids))
            if split == "test":
                self.assertTrue(
                    any(not img["has_defects"] for img in coco["source_images"])
                )
                self.assertEqual(len(source_ids), len(coco["source_images"]))
                for img in coco["source_images"]:
                    self.assertTrue((output / split / img["file_name"]).is_file())
                    self.assertEqual((img["width"], img["height"]), (800, 64))
        self.assertEqual(
            read_annotations(source / "_annotations.coco.json"), original
        )

    def test_pipeline_rejects_missing_sources_configuration(self):
        for broken in ("data_removed", "data_empty", "data_incomplete"):
            with self.subTest(case=broken):
                if broken == "data_removed":
                    del self.params["data"]
                elif broken == "data_empty":
                    self.params["data"] = {}
                else:
                    self.params["data"] = {
                        "sources": [str(self.sources[0])]
                    }
                with self.assertRaises(ValueError):
                    main([])
                self.assertFalse((self.trainings / "sources").exists())

    def test_tiles_sample_once_per_source_not_per_tile(self):
        from src.preprocess.tile_creation import create_tiles

        source = self.root / "sampling"
        coco = self.make_dataset(source, count=2)
        for img in coco["images"]:
            img.update(width=1000, height=64)
            with Image.new("RGB", (1000, 64), "white") as image:
                image.save(source / img["file_name"])
        self.params["preprocess"]["resize"] = 1.0
        self.params["preprocess"]["saving-prob"] = 0.5
        with patch("src.preprocess.tile_creation.random.Random") as random_class:
            random_class.return_value.random.side_effect = [0.2, 0.8]
            with redirect_stdout(io.StringIO()):
                create_tiles(self.root / "tiles", source, annotations=coco)
            self.assertEqual(random_class.return_value.random.call_count, 2)
        tiles = read_annotations(self.root / "tiles" / "_annotations.coco.json")
        for stem, expected in (("image0_", 3), ("image1_", 1)):
            self.assertEqual(
                sum(img["file_name"].startswith(stem) for img in tiles["images"]),
                expected,
            )

    def test_tiles_preserve_healthy_dark_sources_when_saving_all(self):
        from src.preprocess.tile_creation import create_tiles

        source = self.root / "healthy"
        coco = self.make_dataset(source, count=1)
        coco["annotations"] = []
        with Image.new("RGB", (800, 64), "black") as image:
            image.save(source / "image0.png")
        self.params["preprocess"]["resize"] = 1.0
        for probability, expected in ((0.0, 0), (1.0, 3)):
            output = self.root / f"probability{probability}"
            with redirect_stdout(io.StringIO()):
                create_tiles(
                    output, source, annotations=coco, saving_prob=probability,
                    save_source_images=True,
                )
            result = read_annotations(output / "_annotations.coco.json")
            self.assertEqual(len(result["images"]), expected)
            self.assertEqual(len(result["source_images"]), 1)
            self.assertFalse(result["source_images"][0]["has_defects"])

    def test_tiles_map_roi_offsets_into_complete_resized_image(self):
        from src.preprocess.roi import ROI
        from src.preprocess.tile_creation import create_tiles

        source = self.root / "roi"
        coco = self.make_dataset(source, count=1)
        with Image.new("RGB", (1000, 500), "white") as image:
            image.save(source / "image0.png")
        coco["annotations"][0].update(
            bbox=[120, 60, 20, 16],
            segmentation=[[120, 60, 140, 60, 140, 76, 120, 76]],
        )
        self.params["preprocess"]["apply-roi"] = True
        with patch(
            "src.preprocess.tile_creation.detect_roi",
            return_value=ROI(100, 50, 400, 200),
        ):
            with redirect_stdout(io.StringIO()):
                create_tiles(
                    self.root / "roi_tiles", source, annotations=coco,
                    saving_prob=1.0, save_source_images=True,
                )
        result = read_annotations(self.root / "roi_tiles" / "_annotations.coco.json")
        self.assertEqual(result["images"][0]["tile_origin"], [50, 25])
        self.assertEqual(result["annotations"][0]["bbox"], [10, 5, 10, 8])
        full = result["source_images"][0]
        self.assertEqual((full["width"], full["height"]), (500, 250))
        self.assertEqual(full["annotations"][0]["bbox"], [60, 30, 10, 8])

    def test_augmentation_handles_empty_split(self):
        source = self.root / "empty"
        self.make_dataset(source, count=0)
        with redirect_stdout(io.StringIO()):
            output = augment_dataset(source)
        self.assertEqual(read_annotations(output)["images"], [])

    def test_augmentation_preserves_geometry_and_is_reproducible(self):
        source = self.root / "images"
        original = self.make_dataset(source, count=1)
        state = random.getstate()
        with redirect_stdout(io.StringIO()):
            first = augment_dataset(source, self.root / "aug1")
            second = augment_dataset(source, self.root / "aug2")
        self.assertEqual(random.getstate(), state)
        first_coco = read_annotations(first)
        self.assertEqual(first_coco, read_annotations(second))
        augmented = first_coco["images"][1]
        self.assertEqual(
            (first.parent / augmented["file_name"]).read_bytes(),
            (second.parent / augmented["file_name"]).read_bytes(),
        )
        for field in ("bbox", "segmentation", "area", "category_id"):
            self.assertEqual(
                first_coco["annotations"][1][field],
                original["annotations"][0][field],
            )
        self.assertEqual(
            read_annotations(source / "_annotations.coco.json"), original
        )

    def test_split_actions_and_annotation_references(self):
        for action in ("copy", "move", "ignore"):
            with self.subTest(action=action):
                source = self.root / action
                original = self.make_dataset(source)
                with redirect_stdout(io.StringIO()):
                    paths = split_dataset(source, source / "splits", image_action=action)
                image_ids = []
                for path in paths.values():
                    coco = read_annotations(path)
                    ids = {img["id"] for img in coco["images"]}
                    self.assertTrue(
                        all(ann["image_id"] in ids for ann in coco["annotations"])
                    )
                    image_ids.extend(ids)
                    for img in coco["images"]:
                        self.assertEqual(
                            (path.parent / img["file_name"]).exists(),
                            action != "ignore",
                        )
                self.assertCountEqual(
                    image_ids, [img["id"] for img in original["images"]]
                )
                self.assertEqual((source / "image0.png").exists(), action != "move")

    def test_label_correction_clamps_filters_and_remaps(self):
        source = self.root / "labels"
        for split in SPLITS:
            coco = self.make_dataset(source / split, count=1)
            coco["annotations"][0]["bbox"] = [-2, 20, 40, 10]
            tiny = copy.deepcopy(coco["annotations"][0])
            tiny.update(id=2, category_id=7, bbox=[1, 1, 2, 2])
            coco["annotations"].append(tiny)
            coco["images"].append({
                "id": 2, "file_name": "unannotated.png", "width": 32, "height": 24,
            })
            write_annotations(source / split / "_annotations.coco.json", coco)
        with redirect_stdout(io.StringIO()):
            correct_labels(source)
        for split in SPLITS:
            coco = read_annotations(source / split / "_annotations.coco.json")
            self.assertEqual(len(coco["annotations"]), 1)
            self.assertEqual(coco["annotations"][0]["bbox"], [0, 20, 32, 4])
            self.assertEqual(coco["annotations"][0]["category_id"], 0)
            self.assertEqual(coco["categories"], [{"id": 0, "name": "b"}])
            self.assertEqual(len(coco["images"]), 2)

    def test_invalid_ratios_do_not_modify_source(self):
        source = self.sources[0]
        self.params["data"]["sources"] = [str(source)]
        coco = self.make_dataset(source)
        self.params["preprocess"]["train-ratio"] = 0.8
        with self.assertRaises(ValueError):
            main([])
        self.assertEqual(read_annotations(source / "_annotations.coco.json"), coco)
        self.assertFalse((self.trainings / "sources").exists())


if __name__ == "__main__":
    unittest.main()
