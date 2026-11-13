import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from test_incremental_preprocess import GLOBAL, SIZE, make_config, make_source, write_coco

from src.pipelines.cached import lock as lock_module
from src.pipelines.cached import view as view_module
from src.pipelines.cached.cache import TILES_DIR, read_tile_array, source_cache_dir, update_cache
from src.pipelines.cached.dataset import CachedTileDataset
from src.pipelines.cached.lock import CacheLock
from src.pipelines.cached.sampler import EpochNegativeSampler
from src.pipelines.cached.verify import verify_view
from src.pipelines.cached.view import ANNOTATIONS_FILE, build_view, load_view
from src.pipelines.disk.params import CachedConfig
from src.preprocess.annotations import SPLITS, read_annotations

TILES_PER_IMAGE = 9  # 800 px image, nano tile 384 -> 3x3 tiles
IMAGES = 8


@pytest.fixture
def source(tmp_path):
    make_source(tmp_path / "source", count=IMAGES)
    return tmp_path / "source"


def run(source, tmp_path, cfg=None, cached_cfg=None, **kwargs):
    cfg = cfg or make_config()
    cached_cfg = cached_cfg or CachedConfig()
    results = update_cache([source], tmp_path / "cache", cfg, GLOBAL, cached_cfg, **kwargs)
    view, _ = build_view(results, tmp_path / "dataset", getattr(cfg, "classes", None), cached_cfg.tile_format)
    return results, view


def tile_mtimes(tmp_path):
    return {path.name: path.stat().st_mtime_ns for path in (tmp_path / "cache").rglob(f"{TILES_DIR}/*")}


def stats(results):
    return sum((Counter(result.stats) for result in results), Counter())


def test_second_run_and_split_or_class_changes_reuse_every_tile(source, tmp_path):
    results, view = run(source, tmp_path)
    assert stats(results)["tiles_written"] == IMAGES * TILES_PER_IMAGE
    assert sum(view.summary[f"{split}/tiles"] for split in SPLITS) == IMAGES * TILES_PER_IMAGE
    before = tile_mtimes(tmp_path)

    for cfg in (
        make_config(),
        make_config(train_ratio=0.25, val_ratio=0.25, test_ratio=0.5),
        make_config(classes=["dent"]),
    ):
        results, view = run(source, tmp_path, cfg)
        assert stats(results)["tiles_written"] == 0
        assert tile_mtimes(tmp_path) == before
        verify_view(view.dataset_dir, deep=True)


def test_layout_change_rewrites_and_removed_image_deletes_tiles(source, tmp_path):
    run(source, tmp_path)
    results, _ = run(source, tmp_path, make_config(resize=0.5))
    assert stats(results)["tiles_written"] == IMAGES * 4  # 400 px -> 2x2 tiles of 384 (with overlap)

    coco = read_annotations(source / "_annotations.coco.json")
    removed = coco["images"].pop()
    coco["annotations"] = [ann for ann in coco["annotations"] if ann["image_id"] != removed["id"]]
    write_coco(source, coco)
    results, view = run(source, tmp_path, make_config(resize=0.5))
    assert stats(results)["tiles_written"] == 0
    assert stats(results)["deleted"] == 1
    assert len(tile_mtimes(tmp_path)) == (IMAGES - 1) * 4
    assert sum(view.summary[f"{split}/originals"] for split in SPLITS) == IMAGES - 1


@pytest.mark.parametrize("tile_format", ["png", "jpeg", "raw"])
def test_tile_formats_round_trip(source, tmp_path, tile_format):
    results, view = run(source, tmp_path, cached_cfg=CachedConfig(tile_format=tile_format))
    tile = next((tmp_path / "cache").rglob(f"{TILES_DIR}/*"))
    pixels = read_tile_array(tile)
    assert pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] == 3
    if tile_format != "jpeg":
        assert np.all(pixels[..., 1] == 100) and np.all(pixels[..., 2] == 150)

    expected = {"png": ".png", "jpeg": ".jpg", "raw": ".npy"}[tile_format]
    for split in SPLITS:
        coco = read_annotations(view.dataset_dir / split / ANNOTATIONS_FILE)
        suffixes = {Path(image["file_name"]).suffix for image in coco["images"]}
        assert suffixes == {".png" if tile_format == "raw" and split != "train" else expected}
    verify_view(view.dataset_dir, deep=True)


def test_view_hardlinks_tiles_and_falls_back_to_copies(source, tmp_path, monkeypatch):
    _, view = run(source, tmp_path)
    coco = read_annotations(view.dataset_dir / "train" / ANNOTATIONS_FILE)
    cache_tiles = source_cache_dir(tmp_path / "cache", source) / TILES_DIR
    file_name = coco["images"][0]["file_name"]
    assert os.path.samefile(view.dataset_dir / "train" / file_name, cache_tiles / file_name)

    def no_links(*args):
        raise OSError("cross-device link")

    monkeypatch.setattr(view_module.os, "link", no_links)
    _, copied = run(source, tmp_path / "other")
    other_tiles = source_cache_dir(tmp_path / "other" / "cache", source) / TILES_DIR
    destination = copied.dataset_dir / "train" / file_name
    assert not os.path.samefile(destination, other_tiles / file_name)
    assert destination.read_bytes() == (other_tiles / file_name).read_bytes()


def test_classes_filter_annotations_and_test_has_source_images(source, tmp_path):
    _, view = run(source, tmp_path, make_config(classes=["dent"]))
    assert view.categories == [{"id": 1, "name": "dent", "supercategory": "defects"}]
    for split in SPLITS:
        coco = read_annotations(view.dataset_dir / split / ANNOTATIONS_FILE)
        assert all(ann["category_id"] == 1 for ann in coco["annotations"])
        assert sum(image["has_defects"] for image in coco["images"]) == len(coco["annotations"])
        assert bool(coco["source_images"]) == (split == "test")
    test = read_annotations(view.dataset_dir / "test" / ANNOTATIONS_FILE)
    for source_image in test["source_images"]:
        assert (view.dataset_dir / "test" / source_image["file_name"]).is_file()
        assert source_image["width"] == SIZE


def test_load_view_rejects_modified_annotations(source, tmp_path):
    _, view = run(source, tmp_path)
    assert load_view(view.dataset_dir).fingerprint == view.fingerprint

    path = view.dataset_dir / "train" / ANNOTATIONS_FILE
    coco = read_annotations(path)
    coco["annotations"].pop()
    path.write_text(json.dumps(coco), encoding="utf-8")
    with pytest.raises(ValueError, match="no coinciden"):
        load_view(view.dataset_dir)


def test_sampler_keeps_positives_and_resamples_negatives_per_epoch():
    positive = [index % 5 == 0 for index in range(100)]  # 20 positives, 80 negatives
    sampler = EpochNegativeSampler(positive, 0.25, batch_unit=8, min_length=0, seed=1)
    epochs = []
    for epoch in range(2):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        assert len(indices) == len(sampler) == 40
        assert {index for index in range(100) if positive[index]} <= set(indices)
        assert len({index for index in indices if not positive[index]}) == 20
        epochs.append({index for index in indices if not positive[index]})
    assert epochs[0] != epochs[1]
    sampler.set_epoch(0)
    assert {index for index in sampler if not positive[index]} == epochs[0]

    assert len(EpochNegativeSampler(positive, 0.25, batch_unit=6, min_length=50)) == 54


def test_dataset_replaces_corrupt_tiles_up_to_the_limit(source, tmp_path):
    _, view = run(source, tmp_path)
    split_dir = view.dataset_dir / "train"
    dataset = CachedTileDataset(split_dir, view.categories, max_bad_samples=1)
    assert sum(dataset.positive) == view.summary["train/positive_tiles"]
    image, _ = dataset[0]
    assert image.size == (384, 384)

    for image_id in dataset.ids:
        path = split_dir / dataset.coco.imgs[image_id]["file_name"]
        path.unlink()  # never write through the hardlink into the cache
        path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="decodificar"):
        dataset[0]
    assert dataset.bad_samples == 1
    with pytest.raises(RuntimeError):
        verify_view(view.dataset_dir, deep=True)


def test_dataset_custom_augmentation_changes_pixels(source, tmp_path):
    _, view = run(source, tmp_path)
    dataset = CachedTileDataset(view.dataset_dir / "train", view.categories, augment_prob=1.0, max_transforms=3)
    plain = CachedTileDataset(view.dataset_dir / "train", view.categories)
    augmented = [np.asarray(dataset[index][0]) for index in range(4)]
    originals = [np.asarray(plain[index][0]) for index in range(4)]
    assert any(not np.array_equal(a, b) for a, b in zip(augmented, originals, strict=True))


def test_lock_rejects_second_holder_and_takes_over_stale_lock(tmp_path, monkeypatch):
    with CacheLock(tmp_path):
        with pytest.raises(RuntimeError, match="bloqueada"):
            CacheLock(tmp_path).acquire()
    assert not (tmp_path / lock_module.LOCK_NAME).exists()

    with CacheLock(tmp_path) as held:
        held._held = False  # simulate a crash: the file stays behind
    monkeypatch.setattr(lock_module, "_pid_alive", lambda pid: False)
    with CacheLock(tmp_path):
        pass
    assert not (tmp_path / lock_module.LOCK_NAME).exists()


def test_cached_config_validation():
    assert CachedConfig(tile_format="JPG").tile_format == "jpeg"
    with pytest.raises(ValueError):
        CachedConfig(tile_format="tiff")
    with pytest.raises(ValueError):
        CachedConfig(negative_ratio=1.5)

@pytest.mark.parametrize("backend", ["custom", "albumentations"])
def test_datamodule_builds_aligned_train_batches(source, tmp_path, backend):
    from src.pipelines.cached.datamodule import CachedDataModule, augmentation_settings
    from src.pipelines.cached.train import rfdetr_overrides
    from src.pipelines.disk.params import TrainConfig
    from src.pipelines.disk.train import MODEL_CONFIGS, get_rfdetr_train_cfg

    _, view = run(source, tmp_path)
    cached_cfg = CachedConfig(augmentation_backend=backend, num_workers=0, negative_ratio=0.5)
    train_cfg = TrainConfig(lr=1e-4, lr_encoder=1e-5, batch_size=2, grad_accum_steps=2, epochs=1, weight_decay=1e-4,
                            tensorboard=False)
    model_config = MODEL_CONFIGS["nano"](num_classes=len(view.categories))
    rfdetr_cfg = get_rfdetr_train_cfg(view.dataset_dir, tmp_path / "out", train_cfg).model_copy(
        update=rfdetr_overrides(cached_cfg, None)
    )
    _, _, augment_prob = augmentation_settings(cached_cfg)
    data_module = CachedDataModule(model_config, rfdetr_cfg, view, negative_ratio=0.5, augment_prob=augment_prob)
    data_module.setup("fit")

    loader = data_module.train_dataloader()
    assert len(loader.sampler) % 4 == 0
    images, targets = next(iter(loader))
    assert len(targets) == 2
    assert all(int(label) < len(view.categories) for target in targets for label in target["labels"])
    assert len(data_module.val_dataloader().dataset) == view.summary["valid/tiles"]
