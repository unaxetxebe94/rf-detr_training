import copy
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image
from rfdetr.datasets.coco import CocoDetection

from src.core.params import get_preprocess_config
from src.preprocess import run_preprocess as pipeline
from src.preprocess.annotations import SPLITS, read_annotations, write_annotations
from src.preprocess.class_filter import filter_classes
from src.test.categories import (
    align_categories,
    align_door_categories,
    validate_checkpoint_categories,
)
from src.test.predictions import predict_dataset


def make_splits(root):
    for split, ids in zip(SPLITS, ((7, 20), (31, 5), (9, 40)), strict=True):
        annotations = [
            {"id": index, "image_id": index, "category_id": category_id, "bbox": [1, 2, 8, 8]}
            for index, category_id in enumerate(ids, 1)
        ]
        write_annotations(root / split / "_annotations.coco.json", {
            "categories": [
                {"id": ids[0], "name": "Scratch"}, {"id": ids[1], "name": "DENT"},
            ],
            "images": [{"id": 1}, {"id": 2}],
            "annotations": annotations,
            "source_images": [
                {"id": index, "annotations": [copy.deepcopy(ann)], "has_defects": True}
                for index, ann in enumerate(annotations, 1)
            ],
        })


def test_class_filter_keeps_negatives_and_remaps_all_splits_and_originals(tmp_path):
    make_splits(tmp_path)
    filter_classes(tmp_path, ["sCrAtCh"])
    for split in SPLITS:
        coco = read_annotations(tmp_path / split / "_annotations.coco.json")
        assert coco["categories"] == [{"id": 1, "name": "scratch"}]
        assert coco["images"] == [{"id": 1}, {"id": 2}]
        assert coco["annotations"] == [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 8, 8]},
        ]
        positive, negative = coco["source_images"]
        assert positive["annotations"] == coco["annotations"]
        assert positive["has_defects"] is True
        assert negative == {"id": 2, "annotations": [], "has_defects": False}


@pytest.mark.parametrize("classes", [None, ["dent", "SCRATCH"]])
def test_all_classes_and_configured_order_have_consistent_label_ids(tmp_path, classes):
    make_splits(tmp_path)
    filter_classes(tmp_path, classes)
    names = ["scratch", "dent"] if classes is None else ["dent", "scratch"]
    for split in SPLITS:
        coco = read_annotations(tmp_path / split / "_annotations.coco.json")
        assert coco["categories"] == [
            {"id": index, "name": name} for index, name in enumerate(names, 1)
        ]
        assert [ann["category_id"] for ann in coco["annotations"]] == (
            [1, 2] if classes is None else [2, 1]
        )


@pytest.mark.parametrize("classes", [
    [], "scratch", [1], [None], [""], ["  "], ["Scratch", "SCRATCH"], ["missing"],
])
def test_invalid_class_selection_does_not_partially_rewrite_splits(tmp_path, classes):
    make_splits(tmp_path)
    paths = [tmp_path / split / "_annotations.coco.json" for split in SPLITS]
    original = [path.read_bytes() for path in paths]
    with pytest.raises(ValueError, match="preprocess.classes"):
        filter_classes(tmp_path, classes)
    assert [path.read_bytes() for path in paths] == original


@pytest.fixture
def configured_pipeline(tmp_path, monkeypatch):
    sources = [tmp_path / "alpha", tmp_path / "beta"]
    root = tmp_path / "prepared"
    params = {
        "experiment-name": "test",
        "model-type": "nano",
        "seed": 42,
        "data": {"sources": sources, "training-dataset-path": root},
        "preprocess": {
            "saving-prob": 1.0,
            "min-retained-dimension-ratio": 0.1,
            "resize": 1.0,
            "apply-roi": False,
            "train-ratio": 0.5,
            "val-ratio": 0.25,
            "test-ratio": 0.25,
            "augmentations-per-image": 0,
            "max-transforms-per-sample": 1,
            "n-jobs": 1,
        },
    }
    monkeypatch.setattr("src.core.params.read_params", lambda: params)
    # Keep config patches visible while exercising the real source transformations.
    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", lambda: ThreadPoolExecutor(max_workers=1))
    return sources, root, params


def make_raw_source(directory, second_class):
    directory.mkdir()
    coco = {
        "categories": [{"id": 7, "name": "Scratch"}, {"id": 20, "name": second_class}],
        "images": [], "annotations": [],
    }
    for index in range(8):
        filename = f"image{index}.png"
        with Image.new("RGB", (384, 384), "white") as image:
            image.save(directory / filename)
        coco["images"].append({
            "id": index + 1, "file_name": filename, "width": 384, "height": 384,
        })
        coco["annotations"].append({
            "id": index + 1, "image_id": index + 1,
            "category_id": 7 if index % 2 else 20,
            "bbox": [4, 4, 16, 16], "area": 256, "iscrowd": 0,
            "segmentation": [[4, 4, 20, 4, 20, 20, 4, 20]],
        })
    write_annotations(directory / "_annotations.coco.json", coco)


@pytest.mark.parametrize("selection, names", [
    (None, ["scratch", "dent", "hole"]),
    (["sCrAtCh"], ["scratch"]),
    (["HOLE"], ["hole"]),
])
def test_preprocessing_fuses_only_current_sources_before_filtering(
    configured_pipeline, selection, names,
):
    sources, root, params = configured_pipeline
    if selection is not None:
        params["preprocess"]["classes"] = selection
    assert get_preprocess_config().classes == selection
    for source, second_class in zip(sources, ("DENT", "Hole"), strict=True):
        make_raw_source(source, second_class)
    make_splits(root / "sources" / "old_source9")
    originals = [(source / "_annotations.coco.json").read_bytes() for source in sources]

    assert pipeline.run_preprocess() == 0

    total_images = 0
    for split in SPLITS:
        coco = read_annotations(root / split / "_annotations.coco.json")
        assert coco["categories"] == [
            {"id": index, "name": name} for index, name in enumerate(names, 1)
        ]
        total_images += len(coco["images"])
        assert len({image["file_name"] for image in coco["images"]}) == len(coco["images"])
        assert all((root / split / image["file_name"]).is_file() for image in coco["images"])
        assert all(ann["category_id"] in range(1, len(names) + 1) for ann in coco["annotations"])
        if selection:
            assert len({ann["image_id"] for ann in coco["annotations"]}) < len(coco["images"])
    assert total_images == 16
    assert [(source / "_annotations.coco.json").read_bytes() for source in sources] == originals
    unfiltered = read_annotations(root / "sources" / "alpha_source0" / "train" / "_annotations.coco.json")
    assert [cat["name"] for cat in unfiltered["categories"]] == ["Scratch", "DENT"]
    test = read_annotations(root / "test" / "_annotations.coco.json")
    assert len(test["source_images"]) == 4
    for source in test["source_images"]:
        assert source["has_defects"] == bool(source["annotations"])
        assert (root / "test" / source["file_name"]).is_file()
    assert align_door_categories(test, sources)["source_images"] == test["source_images"]

    training = CocoDetection(
        root / "train", root / "train" / "_annotations.coco.json",
        transforms=None, remap_category_ids=True,
    )
    assert training.label2cat == {index: index + 1 for index in range(len(names))}
    for index in range(len(training)):
        image, target = training[index]
        image.close()
        assert all(0 <= label < len(names) for label in target["labels"].tolist())

    model = Mock()
    model.predict.return_value = SimpleNamespace(xyxy=[[4, 4, 20, 20]], class_id=[0], confidence=[0.9])
    predictions = predict_dataset(model, root / "test", test)
    assert all(prediction[0][1] == 1 for prediction in predictions.values())


def test_worker_failure_prevents_fusion_and_filtering(configured_pipeline, monkeypatch):
    def fail(*args):
        raise RuntimeError("broken source")

    monkeypatch.setattr(pipeline, "_process_single_source", fail)
    fusion, filtering = Mock(), Mock()
    monkeypatch.setattr(pipeline, "_fuse_splitted_datasets", fusion)
    monkeypatch.setattr(pipeline, "filter_classes", filtering)
    assert pipeline.run_preprocess() == 1
    fusion.assert_not_called()
    filtering.assert_not_called()


def test_checkpoint_and_dataset_alignment_ignore_category_name_case():
    categories = [{"id": 1, "name": "scratch"}]
    validate_checkpoint_categories(SimpleNamespace(model=SimpleNamespace(class_names=["SCRATCH"])), categories)
    aligned = align_categories({
        "categories": [{"id": 40, "name": "Scratch"}],
        "annotations": [{"category_id": 40}],
        "source_images": [{"annotations": [{"category_id": 40}], "has_defects": True}],
    }, categories)
    assert aligned["annotations"] == [{"category_id": 1}]
    assert aligned["source_images"][0]["annotations"] == [{"category_id": 1}]
    with pytest.raises(ValueError, match="unique"):
        align_categories({"categories": [], "annotations": []}, categories + [{"id": 2, "name": "SCRATCH"}])
