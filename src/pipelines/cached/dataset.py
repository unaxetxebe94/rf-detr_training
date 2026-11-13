"""RF-DETR dataset over the cached view: fast tile decode, online custom augmentation and bad-tile tolerance."""

import logging
import random
from pathlib import Path

from PIL import Image
from rfdetr.datasets.coco import CocoDetection

from ...preprocess.image_transforms import ALL_AUGMENTATIONS, apply_transforms
from .cache import read_tile_array

logger = logging.getLogger(__name__)


class CachedTileDataset(CocoDetection):
    """One item = one tile of a view split (``.png``/``.jpg`` via OpenCV, ``.npy`` via ``np.load``).

    With ``augment_prob > 0`` every read applies, with that probability, 1..``max_transforms`` random photometric
    transforms (the custom ones of the disk pipeline) before the RF-DETR transforms. With ``max_bad_samples > 0`` an
    undecodable tile is logged and replaced by another random tile, up to that many per process; beyond it (or with
    ``max_bad_samples == 0``) the error is raised.
    """

    def __init__(
        self,
        split_dir: Path,
        categories: list[dict],
        transforms=None,
        *,
        augment_prob: float = 0.0,
        max_transforms: int = 1,
        max_bad_samples: int = 0,
    ) -> None:
        split_dir = Path(split_dir)
        super().__init__(
            split_dir, split_dir / "_annotations.coco.json", transforms,
            remap_category_ids=True, cat2label={cat["id"]: label for label, cat in enumerate(categories)},
        )
        self.augment_prob = augment_prob
        self.max_transforms = max(1, max_transforms)
        self.max_bad_samples = max_bad_samples
        self.bad_samples = 0
        self.positive = [bool(self.coco.imgToAnns.get(image_id)) for image_id in self.ids]

    def _decode_image(self, image_id: int) -> tuple[Image.Image, tuple[float, float]]:
        path = Path(self.root) / self.coco.imgs[image_id]["file_name"]
        image = Image.fromarray(read_tile_array(path))
        if self.augment_prob and random.random() < self.augment_prob:
            rng = random.Random(random.getrandbits(64))
            count = min(rng.randint(1, self.max_transforms), len(ALL_AUGMENTATIONS))
            augmented = apply_transforms(image, rng.sample(ALL_AUGMENTATIONS, count), rng)
            image.close()
            image = augmented
        return image, (1.0, 1.0)

    def __getitem__(self, idx: int):
        while True:
            try:
                return super().__getitem__(idx)
            except Exception as error:
                if self.bad_samples >= self.max_bad_samples:
                    raise
                self.bad_samples += 1
                file_name = self.coco.imgs[self.ids[idx]]["file_name"]
                logger.warning(
                    f"Tile ilegible {file_name} ({error}); se sustituye por otro "
                    f"({self.bad_samples}/{self.max_bad_samples} en este proceso)."
                )
                idx = random.randrange(len(self.ids))
