from collections import Counter

import pytest

from src.preprocess.annotations import SPLITS
from pipelines.streaming import dataset as dataset_module
from pipelines.streaming.dataset import InMemoryTileDataset
from pipelines.streaming.plan import build_tile_plan
from pipelines.streaming.sampler import OriginAffineBatchSampler
from test_incremental_preprocess import GLOBAL, make_config, make_source

TILES_PER_IMAGE = 9  # 800 px image, nano tile 384 -> 3x3 tiles


@pytest.fixture
def source(tmp_path):
    make_source(tmp_path / "source", count=8)
    return tmp_path / "source"


def test_plan_splits_per_original_and_keeps_disk_labels(source):
    plan = build_tile_plan([source], make_config(classes=None), GLOBAL)

    assert [cat["name"] for cat in plan.categories] == ["scratch", "dent"]
    assert [cat["id"] for cat in plan.categories] == [1, 2]
    paths = [original.path for split in SPLITS for original in plan.splits[split].originals]
    assert len(paths) == len(set(paths)) == 8
    for split in SPLITS:
        tiles_per_origin = Counter(origin for origin, _, _ in plan.splits[split].tiles)
        assert set(tiles_per_origin.values()) <= {TILES_PER_IMAGE}
        anns = [ann for _, _, tile_anns in plan.splits[split].tiles for ann in tile_anns]
        assert len(anns) == len(plan.splits[split].originals)
        assert all(ann["bbox"] == [10, 10, 30, 30] and ann["category_id"] in (1, 2) for ann in anns)


def test_plan_saving_prob_drops_empty_train_tiles_only(source):
    plan = build_tile_plan([source], make_config(saving_prob=0.0, classes=["dent"]), GLOBAL)

    assert plan.categories == [{"id": 1, "name": "dent", "supercategory": "none"}]
    assert all(anns for _, _, anns in plan.splits["train"].tiles)
    assert len(plan.splits["test"].tiles) == TILES_PER_IMAGE * len(plan.splits["test"].originals)


@pytest.mark.parametrize("train", [False, True])
def test_sampler_keeps_each_original_on_one_worker(train):
    origins = [origin for origin in range(7) for _ in range(5 + origin)]
    sampler = OriginAffineBatchSampler(
        origins, 4, num_streams=3, images_per_stream=2, shuffle=train, pad=train, batch_multiple=5,
    )
    batches = list(sampler)

    assert len(batches) == len(sampler)
    worker_of = {}
    for position, batch in enumerate(batches):
        for index in batch:
            assert worker_of.setdefault(origins[index], position % 3) == position % 3
    seen = Counter(index for batch in batches for index in batch)
    assert set(seen) == set(range(len(origins)))
    if train:
        assert len(batches) % 5 == 0 and all(len(batch) == 4 for batch in batches)
    else:
        assert set(seen.values()) == {1}
    assert sampler.epoch == 1


def test_dataset_decodes_each_original_once(source, monkeypatch):
    plan = build_tile_plan([source], make_config(classes=None), GLOBAL)
    loads = Counter()
    original_loader = dataset_module.load_original
    monkeypatch.setattr(dataset_module, "load_original", lambda original: loads.update([original.path]) or original_loader(original))
    dataset = InMemoryTileDataset(plan.splits["train"], plan.categories, augmentations_per_image=1, cache_size=2)
    sampler = OriginAffineBatchSampler(dataset.sample_origins, 3, images_per_stream=2, shuffle=True, pad=True)

    targets = [dataset[index][1] for batch in sampler for index in batch]

    assert len(dataset) == 2 * TILES_PER_IMAGE * len(plan.splits["train"].originals)
    assert set(loads.values()) == {1} and len(loads) == len(plan.splits["train"].originals)
    image, target = dataset[0]
    assert image.size == (384, 384)
    labeled = [target for target in targets if len(target["labels"])]
    assert labeled and all(target["boxes"].tolist() == [[10, 10, 40, 40]] for target in labeled)
