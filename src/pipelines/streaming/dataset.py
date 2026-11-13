"""RF-DETR dataset that crops tiles on the fly from originals kept in memory."""

import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from rfdetr.datasets.coco import CocoDetection, ConvertCoco

from ...preprocess.image import open_image
from ...preprocess.image_transforms import ALL_AUGMENTATIONS, apply_transforms
from ...preprocess.roi import ROI
from ...preprocess.tile_creation import apply_layout
from .plan import Original, SplitPlan


def load_original(original: Original) -> np.ndarray:
    """Decode an original once, with the same ROI crop and resize used to plan its tiles (HxWx3 uint8)."""
    img = open_image(original.path)
    img = apply_layout(img, ROI(*original.roi) if original.roi else None, original.scale)
    if img.interpretation not in ("srgb", "rgb") or img.format != "uchar":
        img = img.colourspace("srgb")
    if img.format != "uchar":
        img = img.cast("uchar")
    if img.bands == 1:
        img = img.bandjoin([img, img])
    elif img.bands > 3:
        img = img.extract_band(0, n=3)
    return np.ndarray(buffer=img.write_to_memory(), dtype=np.uint8, shape=(img.height, img.width, img.bands))


def build_coco(plan: SplitPlan, categories: list[dict]) -> dict:
    images, annotations = [], []
    for image_id, (origin, (x, y, w, h), anns) in enumerate(plan.tiles, 1):
        images.append({
            "id": image_id, "width": w, "height": h,
            "file_name": f"{Path(plan.originals[origin].path).stem}_{plan.originals[origin].source_id}_tile_{x}_{y}",
        })
        for ann in anns:
            annotations.append({"iscrowd": 0, **ann, "id": len(annotations) + 1, "image_id": image_id})
    return {"images": images, "annotations": annotations, "categories": [dict(cat) for cat in categories]}


class InMemoryTileDataset(CocoDetection):
    """One item = one tile; train repeats every tile `1 + augmentations` times (copies > 0 get photometric noise).

    Each process keeps an LRU of the last `cache_size` decoded originals, so feeding all the tiles of an
    original decodes it once as long as the sampler sends its tiles together to the same worker
    (see :class:`OriginAffineBatchSampler`).
    """

    def __init__(
        self,
        plan: SplitPlan,
        categories: list[dict],
        transforms=None,
        *,
        augmentations_per_image: int = 0,
        max_transforms_per_sample: int = 1,
        cache_size: int = 2,
    ) -> None:
        if cache_size < 1:
            raise ValueError("cache_size debe ser al menos 1.")
        coco = COCO()
        coco.dataset = build_coco(plan, categories)
        coco.createIndex()
        # VisionDataset/torchvision CocoDetection attributes, set without reading an annotation file.
        self.root = ""
        self.transform = self.target_transform = self.transforms = None
        self.coco = coco
        self.ids = [image["id"] for image in coco.dataset["images"]]
        self.cat2label = {cat["id"]: label for label, cat in enumerate(categories)}
        self.label2cat = {label: cat_id for cat_id, label in self.cat2label.items()}
        self.coco.label2cat = self.label2cat
        self.include_masks = False
        self.include_keypoints = False
        self._draft_size = None
        self.prepare = ConvertCoco(cat2label=self.cat2label)
        self._transforms = transforms

        self.originals = plan.originals
        self.tiles = [(origin, region) for origin, region, _ in plan.tiles]
        self.copies = 1 + augmentations_per_image
        self.max_transforms = max_transforms_per_sample
        self.cache_size = cache_size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.loads = 0

    def __len__(self) -> int:
        return len(self.tiles) * self.copies

    @property
    def sample_origins(self) -> list[int]:
        """Original image index of every sample index (what the batch sampler groups by)."""
        return [origin for origin, _ in self.tiles] * self.copies

    def __getstate__(self):
        # Workers start with an empty cache instead of pickling decoded images.
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def _original(self, origin: int) -> np.ndarray:
        array = self._cache.get(origin)
        if array is not None:
            self._cache.move_to_end(origin)
            return array
        while len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        array = load_original(self.originals[origin])
        self._cache[origin] = array
        self.loads += 1
        return array

    def load_tile(self, idx: int) -> tuple[Image.Image, int]:
        tile_index, copy_index = idx % len(self.tiles), idx // len(self.tiles)
        origin, (x, y, w, h) = self.tiles[tile_index]
        image = Image.fromarray(np.ascontiguousarray(self._original(origin)[y:y + h, x:x + w]))
        if copy_index:
            rng = random.Random(random.getrandbits(64))
            chosen = rng.sample(ALL_AUGMENTATIONS, min(rng.randint(1, self.max_transforms), len(ALL_AUGMENTATIONS)))
            augmented = apply_transforms(image, chosen, rng)
            image.close()
            image = augmented
        return image, self.ids[tile_index]

    def __getitem__(self, idx: int):
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        img, image_id = self.load_tile(idx)
        target = {"image_id": image_id, "annotations": self._load_target(image_id)}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
