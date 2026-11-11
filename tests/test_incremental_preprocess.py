from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from PIL import Image

from src.preprocess import incremental
from src.preprocess import run_preprocess as pipeline
from src.preprocess.annotations import SPLITS, read_annotations, write_annotations
from src.preprocess.incremental import MANIFEST_NAME, process_source
from src.preprocess.splitter import assign_splits
from src.test.image_evaluation import evaluate_images

SIZE = 800  # Mayor que el tile nano (384): 9 tiles por imagen sin ampliar, solo uno anotado.


def make_config(**overrides):
    values = {
        "saving_prob": 1.0, "min_retained_dimension_ratio": 0.1, "resize": 1.0, "apply_roi": False,
        "train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.25,
        "augmentations_per_image": 1, "max_transforms_per_sample": 2, "n_jobs": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


GLOBAL = SimpleNamespace(seed=42, model_type="nano")


def save_image(path, value):
    Image.new("RGB", (SIZE, SIZE), (value, 100, 150)).save(path)


def annotation(ann_id, image_id, category_id, x=10):
    return {
        "id": ann_id, "image_id": image_id, "category_id": category_id,
        "bbox": [x, 10, 30, 30], "area": 900, "iscrowd": 0,
        "segmentation": [[x, 10, x + 30, 10, x + 30, 40, x, 40]],
    }


def make_source(directory, count=8):
    directory.mkdir(parents=True, exist_ok=True)
    coco = {"categories": [{"id": 7, "name": "Scratch"}, {"id": 20, "name": "Dent"}], "images": [], "annotations": []}
    for index in range(count):
        add_image(directory, coco, index)
    write_annotations(directory / "_annotations.coco.json", coco)
    return coco


def add_image(directory, coco, index):
    file_name = f"image{index}.png"
    save_image(directory / file_name, index)
    image_id = max((image["id"] for image in coco["images"]), default=0) + 1
    coco["images"].append({"id": image_id, "file_name": file_name, "width": SIZE, "height": SIZE})
    ann_id = max((ann["id"] for ann in coco["annotations"]), default=0) + 1
    coco["annotations"].append(annotation(ann_id, image_id, 7 if index % 2 else 20))
    return image_id


def write_coco(directory, coco):
    write_annotations(directory / "_annotations.coco.json", coco)


def splits_by_file(output):
    manifest = read_annotations(output / MANIFEST_NAME)
    return {entry["file_name"]: entry["split"] for entry in manifest["images"].values()}


def output_files(output):
    return {
        path.relative_to(output).as_posix(): path.stat().st_mtime_ns
        for path in output.rglob("*.png")
    }


def assert_consistent(output):
    """Every referenced file exists, no image file is unreferenced and categories match."""
    referenced = set()
    categories = None
    for split in SPLITS:
        coco = read_annotations(output / split / "_annotations.coco.json")
        categories = categories or coco["categories"]
        assert coco["categories"] == categories
        ids = {cat["id"] for cat in categories}
        assert all(ann["category_id"] in ids for ann in coco["annotations"])
        for source in coco["source_images"]:
            assert all(ann["category_id"] in ids for ann in source["annotations"])
        names = [image["file_name"] for image in coco["images"]]
        names += [source["file_name"] for source in coco["source_images"]]
        referenced.update(f"{split}/{name}".replace("\\", "/") for name in names)
    assert referenced == set(output_files(output))


@pytest.fixture
def dataset(tmp_path):
    source, output = tmp_path / "raw", tmp_path / "out"
    coco = make_source(source)
    stats = process_source(source, output, make_config(), GLOBAL)
    assert stats["new"] == 8
    return source, output, coco


def test_unchanged_rerun_opens_no_images_and_keeps_outputs(dataset, monkeypatch):
    source, output, _ = dataset
    jsons = {split: (output / split / "_annotations.coco.json").read_bytes() for split in SPLITS}
    files = output_files(output)

    def fail(*args):
        raise AssertionError("image opened")

    monkeypatch.setattr(incremental, "open_image", fail)
    monkeypatch.setattr(incremental, "augment_image", fail)
    stats = process_source(source, output, make_config(), GLOBAL)

    assert stats["reused"] == 8 and not stats.get("failed") and not stats.get("tiles_written")
    assert {split: (output / split / "_annotations.coco.json").read_bytes() for split in SPLITS} == jsons
    assert output_files(output) == files
    assert_consistent(output)


@pytest.mark.parametrize("apply_roi", [False, True])
def test_upscaled_tiles_evaluate_in_source_coordinates_on_initial_and_cached_runs(
    tmp_path, monkeypatch, apply_roi,
):
    from src.preprocess.roi import ROI

    source, output = tmp_path / "raw", tmp_path / "out"
    coco = make_source(source)
    for image in coco["images"]:
        with Image.new("RGB", (32, 24), "white") as pixels:
            pixels.save(source / image["file_name"])
        image.update(width=32, height=24)
    coco["annotations"] = [
        {
            **ann, "bbox": [8, 6, 8, 8], "area": 64,
            "segmentation": [[8, 6, 16, 6, 16, 14, 8, 14]],
        }
        for ann in coco["annotations"]
    ]
    write_coco(source, coco)
    monkeypatch.setattr("src.preprocess.tile_creation.detect_roi", lambda *args: ROI(4, 2, 24, 20))
    cfg = make_config(resize=0.5, apply_roi=apply_roi, augmentations_per_image=0)

    files = {}
    for cached in (False, True):
        stats = process_source(source, output, cfg, GLOBAL)
        assert not stats.get("failed")
        if cached:
            assert not stats.get("tiles_written")
            assert output_files(output) == files
        test = read_annotations(output / "test" / "_annotations.coco.json")
        predictions = {}
        for ann in test["annotations"]:
            predictions.setdefault(ann["image_id"], []).append((ann["bbox"], ann["category_id"], 0.9))
        binary, detection = evaluate_images(test, predictions, output / "test", tmp_path / "evaluation")
        assert binary["Total TP"] == len(test["source_images"])
        assert detection["mAP@50:95"] == 1.0
        if not cached:
            files = output_files(output)
            for tile in test["images"]:
                tile.pop("tile_to_source_scale")
            write_annotations(output / "test" / "_annotations.coco.json", test)
            monkeypatch.setattr(incremental, "open_image", lambda *args: pytest.fail("cached image opened"))


def test_added_modified_and_deleted_images_only_touch_those_images(dataset):
    source, output, coco = dataset
    before = splits_by_file(output)
    files = output_files(output)

    add_image(source, coco, 8)
    save_image(source / "image1.png", 200)
    coco["annotations"][2]["bbox"] = [12, 10, 30, 30]
    removed = coco["images"].pop(3)
    coco["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] != removed["id"]]
    write_coco(source, coco)

    stats = process_source(source, output, make_config(), GLOBAL)

    assert (stats["new"], stats["modified"], stats["deleted"], stats["reused"]) == (1, 2, 1, 5)
    after = splits_by_file(output)
    assert {name: split for name, split in after.items() if name in before} == {
        name: split for name, split in before.items() if name != "image3.png"
    }
    changed = {name for name, mtime in output_files(output).items() if files.get(name) != mtime}
    assert changed and all("image8_" in name or "image1_" in name or "/source_images/" in name for name in changed)
    assert not any("image3_" in name for name in output_files(output))
    assert_consistent(output)


def test_new_category_is_consistent_across_splits(dataset):
    source, output, coco = dataset
    coco["categories"].append({"id": 99, "name": "Hole"})
    coco["annotations"].append(annotation(100, coco["images"][0]["id"], 99, x=200))
    write_coco(source, coco)

    stats = process_source(source, output, make_config(), GLOBAL)

    assert stats["modified"] == 1 and not stats.get("tiles_written")
    categories = read_annotations(output / "train" / "_annotations.coco.json")["categories"]
    assert [cat["name"] for cat in categories] == ["Scratch", "Dent", "Hole"]
    assert_consistent(output)


def test_parameter_changes_only_regenerate_affected_outputs(dataset):
    source, output, _ = dataset
    stats = process_source(source, output, make_config(augmentations_per_image=2), GLOBAL)
    assert not stats.get("tiles_written") and stats["augmentations_written"] > 0
    assert_consistent(output)

    stats = process_source(source, output, make_config(augmentations_per_image=2, saving_prob=0.0), GLOBAL)
    assert stats["tiles_deleted"] > 0 and not stats.get("tiles_written")
    assert_consistent(output)

    stats = process_source(source, output, make_config(augmentations_per_image=2, resize=0.98), GLOBAL)
    assert stats["reused"] == 8 and stats["tiles_written"] > 0
    assert_consistent(output)


def test_corrupt_manifest_or_full_flag_rebuilds_from_scratch(dataset):
    source, output, _ = dataset
    (output / "train" / "stale.png").write_bytes(b"x")
    stats = process_source(source, output, make_config(), GLOBAL, full=True)
    assert stats["new"] == 8
    assert not (output / "train" / "stale.png").exists()

    (output / MANIFEST_NAME).write_text("{broken", encoding="utf-8")
    assert process_source(source, output, make_config(), GLOBAL)["new"] == 8
    assert_consistent(output)


def test_single_new_image_goes_to_the_split_furthest_below_its_ratio():
    coco = {
        "categories": [{"id": 1, "name": "a"}],
        "images": [{"id": index} for index in range(1, 6)],
        "annotations": [{"id": index, "image_id": index, "category_id": 1} for index in range(1, 6)],
    }
    frozen = {1: "train", 2: "train", 3: "valid", 4: "valid"}
    assignments = assign_splits(coco, (0.5, 0.25, 0.25), 42, frozen)
    assert {key: assignments[key] for key in frozen} == frozen
    assert assignments[5] == "test"


def test_fusion_copies_only_changes_and_removes_deleted_images(tmp_path, monkeypatch):
    source, root = tmp_path / "alpha", tmp_path / "prepared"
    coco = make_source(source)
    params = {
        "experiment-name": "test", "model-type": "nano", "seed": 42,
        "data": {"sources": [source], "training-dataset-path": root},
        "preprocess": {
            "saving-prob": 1.0, "min-retained-dimension-ratio": 0.1, "resize": 1.0, "apply-roi": False,
            "train-ratio": 0.5, "val-ratio": 0.25, "test-ratio": 0.25,
            "augmentations-per-image": 0, "max-transforms-per-sample": 1, "n-jobs": 1,
        },
    }
    monkeypatch.setattr("src.core.params.read_params", lambda: params)
    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", lambda: ThreadPoolExecutor(max_workers=1))
    assert pipeline.run_preprocess() == 0
    fused = {
        path.relative_to(root).as_posix(): path.stat().st_mtime_ns
        for split in SPLITS for path in (root / split).rglob("*.png")
    }

    removed = coco["images"].pop(0)
    coco["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] != removed["id"]]
    write_coco(source, coco)
    assert pipeline.run_preprocess() == 0

    remaining = {
        path.relative_to(root).as_posix(): path.stat().st_mtime_ns
        for split in SPLITS for path in (root / split).rglob("*.png")
    }
    assert remaining and not any("image0_" in name for name in remaining)
    assert all(fused[name] == mtime for name, mtime in remaining.items())
    for split in SPLITS:
        fused_coco = read_annotations(root / split / "_annotations.coco.json")
        names = {image["file_name"] for image in fused_coco["images"]}
        names |= {source["file_name"].replace("\\", "/") for source in fused_coco["source_images"]}
        assert names == {name.split("/", 1)[1] for name in remaining if name.startswith(f"{split}/")}
