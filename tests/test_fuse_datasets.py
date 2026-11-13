import tempfile
import unittest
from pathlib import Path

from pipelines.disk.fuse_datasets import (
    _discover_preprocessed_datasets,
    _fuse_datasets,
    _fuse_splitted_datasets,
    _source_sort_key,
)
from src.preprocess.annotations import read_annotations, write_annotations


def make_coco(directory, prefix, categories, category_id, n_tiles=2, source_images=None):
    """Crea un mini dataset preprocesado con imágenes y anotaciones."""
    directory.mkdir(parents=True, exist_ok=True)
    coco = {
        "info": {"creator": "test"},
        "categories": categories,
        "images": [],
        "annotations": [],
    }
    if source_images is not None:
        coco["source_images"] = source_images
    for idx in range(n_tiles):
        name = f"{prefix}_tile{idx}.png"
        (directory / name).write_bytes(f"fake-{prefix}-{idx}".encode())
        coco["images"].append({
            "id": idx + 1,
            "file_name": name,
            "width": 32,
            "height": 32,
        })
        coco["annotations"].append({
            "id": idx + 1,
            "image_id": idx + 1,
            "category_id": category_id,
            "bbox": [2, 2, 8, 8],
        })
    write_annotations(directory / "_annotations.coco.json", coco)
    return directory


class FuseDatasetsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_fuses_more_than_two_datasets(self):
        d1 = self.make_dataset("a", [{"id": 1, "name": "x"}, {"id": 2, "name": "y"}], 1)
        d2 = self.make_dataset("b", [{"id": 5, "name": "y"}, {"id": 6, "name": "z"}], 6)
        d3 = make_coco(
            self.root / "c", "c", [{"id": 2, "name": "x"}, {"id": 9, "name": "w"}], 9,
            source_images=[{
                "id": "door", "file_name": "source_images/door.png",
                "annotations": [{"category_id": 2}, {"category_id": 9}],
            }],
        )
        (d3 / "source_images").mkdir()
        (d3 / "source_images" / "door.png").write_bytes(b"door")
        output = self.root / "fused"
        _fuse_datasets([d1, d2, d3], output)

        result = read_annotations(output / "_annotations.coco.json")
        names = {cat["name"]: cat["id"] for cat in result["categories"]}
        # d1 conserva sus IDs, las demás categorías nuevas se renumeran en orden
        self.assertEqual(names, {"x": 1, "y": 2, "z": 3, "w": 4})
        self.assertEqual(len(result["images"]), 6)
        self.assertEqual(
            [img["file_name"] for img in result["images"]],
            ["a_tile0.png", "a_tile1.png", "b_tile0.png", "b_tile1.png", "c_tile0.png", "c_tile1.png"],
        )
        # Las anotaciones siguen su imagen y su categoría remapeada:
        # d1 anota x->1, d2 anota z->3, d3 anota w->4
        self.assertEqual(
            [ann["category_id"] for ann in result["annotations"]], [1, 1, 3, 3, 4, 4]
        )
        self.assertEqual(
            [ann["image_id"] for ann in result["annotations"]], [1, 2, 3, 4, 5, 6]
        )
        # source_images de d3 se remapea a la categoría acumulada
        source = result["source_images"][0]
        self.assertEqual(
            [ann["category_id"] for ann in source["annotations"]], [1, 4]
        )
        # Copia las imágenes de todos los datasets, incluidas source_images
        for suffix in ("a_tile0.png", "b_tile1.png", "c_tile1.png"):
            self.assertTrue((output / suffix).is_file())
        self.assertTrue((output / "source_images" / "door.png").is_file())

    def test_same_ids_and_categories_two_way_as_before(self):
        d1 = self.make_dataset("a", [{"id": 1, "name": "x"}, {"id": 2, "name": "y"}], 1)
        d2 = self.make_dataset("b", [{"id": 5, "name": "y"}, {"id": 6, "name": "z"}], 6)
        output = self.root / "fused"
        _fuse_datasets([d1, d2], output)
        result = read_annotations(output / "_annotations.coco.json")
        names = {cat["name"]: cat["id"] for cat in result["categories"]}
        self.assertEqual(names, {"x": 1, "y": 2, "z": 3})
        self.assertEqual(len(result["images"]), 4)
        self.assertEqual(len(result["annotations"]), 4)

    def test_category_names_merge_case_insensitively_including_base_aliases(self):
        d1 = make_coco(
            self.root / "a", "a",
            [{"id": 3, "name": "Scratch"}, {"id": 9, "name": "SCRATCH"}], 9,
            source_images=[{"id": "a", "annotations": [{"category_id": 9}]}],
        )
        d2 = make_coco(
            self.root / "b", "b", [{"id": 20, "name": "sCrAtCh"}], 20,
            source_images=[{"id": "b", "annotations": [{"category_id": 20}]}],
        )
        output = self.root / "fused"
        _fuse_datasets([d1, d2], output)
        result = read_annotations(output / "_annotations.coco.json")
        self.assertEqual(result["categories"], [{"id": 3, "name": "scratch"}])
        self.assertEqual([ann["category_id"] for ann in result["annotations"]], [3] * 4)
        self.assertEqual(
            [source["annotations"][0]["category_id"] for source in result["source_images"]], [3, 3]
        )

    def test_single_dataset_only_validates_and_copies(self):
        d1 = self.make_dataset("a", [{"id": 3, "name": "x"}], 3)
        output = self.root / "fused"
        _fuse_datasets([d1], output)
        result = read_annotations(output / "_annotations.coco.json")
        self.assertEqual(result["categories"], [{"id": 3, "name": "x"}])
        self.assertEqual([img["id"] for img in result["images"]], [1, 2])
        self.assertTrue((output / "a_tile0.png").is_file())

    def test_empty_list_and_missing_annotation_file_are_rejected(self):
        with self.assertRaises(ValueError):
            _fuse_datasets([], self.root / "out")
        d1 = self.make_dataset("a", [{"id": 1, "name": "x"}], 1)
        with self.assertRaises(FileNotFoundError):
            _fuse_datasets([d1, self.root / "missing"], self.root / "out")

    def test_fuses_every_split_of_every_dataset(self):
        paths = []
        for name, category in (("a", {"id": 1, "name": "shared"}), ("b", {"id": 2, "name": "second"})):
            dataset = self.root / name
            for split in ("train", "valid", "test"):
                make_coco(
                    dataset / split, f"{name}_{split}",
                    [category], category["id"], n_tiles=1,
                )
            paths.append(dataset)
        output = self.root / "fused"
        _fuse_splitted_datasets(paths, output)
        for split in ("train", "valid", "test"):
            coco = read_annotations(output / split / "_annotations.coco.json")
            self.assertEqual(len(coco["images"]), 2)
            self.assertEqual(
                {ann["category_id"] for ann in coco["annotations"]}, {1, 2}
            )
            for tile in coco["images"]:
                self.assertTrue((output / split / tile["file_name"]).is_file())

    def test_discovers_preprocessed_sources_sorted_by_index(self):
        sources = self.root / "trainings" / "sources"
        for name in (
            "puerta2_source2", "puerta1_source1",
            "puerta10_source10", "not_a_source", "raro_sourcexyz",
        ):
            (sources / name / "train").mkdir(parents=True)
        discovered = [
            path.name for path in _discover_preprocessed_datasets(sources.parent)
        ]
        # Solo dirs *_source<i> con índice numérico, en orden numérico
        self.assertEqual(
            discovered,
            ["puerta1_source1", "puerta2_source2", "puerta10_source10"],
        )
        self.assertEqual(_source_sort_key(Path("x_source7"))[1], 7)

    def test_discovers_requires_sources_directory(self):
        with self.assertRaises(FileNotFoundError):
            _discover_preprocessed_datasets(self.root / "does-not-exist")
        (self.root / "empty_sources").mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "source"):
            _discover_preprocessed_datasets(self.root / "empty_sources")

    def make_dataset(self, name, categories, category_id):
        return make_coco(self.root / name, name, categories, category_id)


if __name__ == "__main__":
    unittest.main()
