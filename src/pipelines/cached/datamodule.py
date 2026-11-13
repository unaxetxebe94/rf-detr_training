"""RF-DETR DataModule over the cached view: cached tile datasets + per-epoch negative sampling."""

import logging
from typing import Any

from rfdetr.training import RFDETRDataModule
from rfdetr.training.module_data import _MIN_TRAIN_BATCHES, _worker_init_fn
from torch.utils.data import DataLoader

from ..streaming.datamodule import build_transforms, dataset_namespace
from .dataset import CachedTileDataset
from .sampler import EpochNegativeSampler
from .view import DatasetView

logger = logging.getLogger(__name__)


def default_aug_config(augment_prob: float) -> dict:
    """Albumentations/Kornia equivalent of the custom photometric augmentations (+ RF-DETR's default flip)."""
    return {
        "HorizontalFlip": {"p": 0.5},
        "ColorJitter": {"brightness": 0.4, "contrast": 0.4, "saturation": 0.5, "hue": 30 / 360, "p": augment_prob},
        "GaussianBlur": {"blur_limit": 7, "p": augment_prob / 5},
    }


def augmentation_settings(cached_cfg) -> tuple[str, dict | None, float]:
    """(RF-DETR ``augmentation_backend``, ``aug_config``, probability of the custom per-tile augmentation)."""
    if cached_cfg.augmentation_backend == "custom":
        # torchvision with aug_config=None keeps RF-DETR's default horizontal flip; photometric ones run in the dataset.
        return "torchvision", None, cached_cfg.augment_prob
    aug_config = cached_cfg.aug_config if cached_cfg.aug_config is not None else default_aug_config(
        cached_cfg.augment_prob
    )
    return cached_cfg.augmentation_backend, aug_config, 0.0


class CachedDataModule(RFDETRDataModule):
    """Injects :class:`CachedTileDataset` splits; `RFDETRDataModule.setup` keeps the backend checks and Kornia setup."""

    def __init__(
        self, model_config, train_config, view: DatasetView, *, negative_ratio: float = 1.0,
        augment_prob: float = 0.0, max_transforms_per_sample: int = 1, max_bad_samples: int = 0, seed: int = 0,
    ) -> None:
        super().__init__(model_config, train_config)
        self.view = view
        self.negative_ratio = negative_ratio
        self.augment_prob = augment_prob
        self.max_transforms_per_sample = max_transforms_per_sample
        self.max_bad_samples = max_bad_samples
        self.seed = seed

    def _build(self, image_set: str, split: str, ns) -> CachedTileDataset:
        train = image_set == "train"
        return CachedTileDataset(
            self.view.dataset_dir / split,
            self.view.categories,
            build_transforms(image_set, self.model_config, ns),
            augment_prob=self.augment_prob if train else 0.0,
            max_transforms=self.max_transforms_per_sample,
            max_bad_samples=self.max_bad_samples if train else 0,
        )

    def setup(self, stage: str) -> None:
        if self._dataset_train is None:
            ns = dataset_namespace(self.model_config, self.train_config)
            self._dataset_train = self._build("train", "train", ns)
            self._dataset_val = self._build("val", "valid", ns)
            self._dataset_test = self._build("test", "test", ns)
        super().setup(stage)

    def train_dataloader(self) -> DataLoader[Any]:
        world_size = getattr(self.trainer, "world_size", 1) if self.trainer else 1
        if world_size > 1:
            logger.warning("Con varios procesos (DDP) no se muestrean negativos por epoch: se usan todos los tiles.")
            return super().train_dataloader()
        dataset: CachedTileDataset = self._require_dataset(self._dataset_train, "fit")
        batch_size = self._resolve_batch_size()
        batch_unit = batch_size * self._effective_grad_accum_steps()
        sampler = EpochNegativeSampler(
            dataset.positive, self.negative_ratio,
            batch_unit=batch_unit, min_length=batch_unit * _MIN_TRAIN_BATCHES, seed=self.seed,
        )
        logger.info(
            f"Train por epoch: {len(sampler.positives)} tiles con defectos + {sampler.negatives_per_epoch}/"
            f"{len(sampler.negatives)} sin defectos -> {len(sampler)} muestras."
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=self._collate_fn_train,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            prefetch_factor=self._prefetch_factor,
            worker_init_fn=_worker_init_fn,
        )
