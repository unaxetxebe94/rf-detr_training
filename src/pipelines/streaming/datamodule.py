"""RF-DETR DataModule fed by in-memory tile datasets instead of a dataset folder."""

from typing import Any

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import AugmentationBackend, MultiScale
from rfdetr.datasets.aug_configs import AUG_CONFIG
from rfdetr.datasets.coco import make_coco_transforms, make_coco_transforms_square_div_64
from rfdetr.datasets.kornia_transforms import is_gpu_postprocess, resolve_augmentation_backend, resolve_backend_for_build
from rfdetr.training import RFDETRDataModule
from rfdetr.training.module_data import _has_cuda_device, _worker_init_fn
from torch.utils.data import DataLoader

from .dataset import InMemoryTileDataset
from .plan import TilePlan
from .sampler import OriginAffineBatchSampler

RFDETR_SPLITS = {"train": "train", "val": "valid", "test": "test"}


class StreamingDataModule(RFDETRDataModule):
    """Injects :class:`InMemoryTileDataset` splits and serves them with worker-affine batch samplers.

    `RFDETRDataModule.setup` only builds the datasets that are still ``None``, so assigning ours first
    keeps the rest of its logic (augmentation backend, class-name checks, Kornia pipeline, collate).
    """

    def __init__(
        self, model_config, train_config, plan: TilePlan, *,
        augmentations_per_image: int = 0, max_transforms_per_sample: int = 1,
        images_per_worker: int = 2, seed: int = 0,
    ) -> None:
        super().__init__(model_config, train_config)
        self.plan = plan
        self.augmentations_per_image = augmentations_per_image
        self.max_transforms_per_sample = max_transforms_per_sample
        self.images_per_worker = images_per_worker
        self.seed = seed

    def _dataset_namespace(self):
        """Same augmentation-backend resolution as ``RFDETRDataModule.setup('fit')`` before building datasets."""
        ns = _namespace_from_configs(self.model_config, self.train_config)
        requested = self.train_config.augmentation_backend
        resolved = resolve_augmentation_backend(requested, has_cuda=_has_cuda_device())
        if resolved != requested:
            ns.augmentation_backend = "kornia" if is_gpu_postprocess(resolved) else "cpu"
        if resolved == AugmentationBackend.ALBU and ns.aug_config is None:
            ns.aug_config = AUG_CONFIG
        return ns

    def _transforms(self, image_set: str, ns):
        multi_scale = MultiScale.from_value(ns.multi_scale)
        builder = make_coco_transforms_square_div_64 if ns.square_resize_div_64 else make_coco_transforms
        return builder(
            image_set,
            self.model_config.resolution,
            multi_scale=multi_scale is not MultiScale.OFF,
            expanded_scales=ns.expanded_scales,
            skip_random_resize=multi_scale is not MultiScale.PER_SAMPLE,
            patch_size=ns.patch_size,
            num_windows=ns.num_windows,
            aug_config=getattr(ns, "aug_config", None),
            scale_jitter=getattr(ns, "scale_jitter", True),
            gpu_postprocess=is_gpu_postprocess(resolve_backend_for_build(getattr(ns, "augmentation_backend", "cpu"))),
            keypoint_flip_pairs=None,
        )

    def _build(self, image_set: str, ns) -> InMemoryTileDataset:
        train = image_set == "train"
        return InMemoryTileDataset(
            self.plan.splits[RFDETR_SPLITS[image_set]],
            self.plan.categories,
            self._transforms(image_set, ns),
            augmentations_per_image=self.augmentations_per_image if train else 0,
            max_transforms_per_sample=self.max_transforms_per_sample,
            cache_size=self.images_per_worker if train else 1,
        )

    def setup(self, stage: str) -> None:
        if self._dataset_train is None:
            ns = self._dataset_namespace()
            self._dataset_train = self._build("train", ns)
            self._dataset_val = self._build("val", ns)
            self._dataset_test = self._build("test", ns)
        super().setup(stage)

    def _loader(self, dataset: InMemoryTileDataset, batch_size: int, *, train: bool) -> DataLoader[Any]:
        # One stream per worker; more workers than originals would leave workers without their own images.
        num_workers = min(self._num_workers, len(dataset.originals))
        sampler = OriginAffineBatchSampler(
            dataset.sample_origins,
            batch_size,
            num_streams=max(1, num_workers),
            images_per_stream=dataset.cache_size,
            shuffle=train,
            pad=train,
            batch_multiple=self._effective_grad_accum_steps() if train else 1,
            seed=self.seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self._collate_fn_train if train else self._collate_fn,
            num_workers=num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers and num_workers > 0,
            prefetch_factor=self._prefetch_factor if num_workers > 0 else None,
            worker_init_fn=_worker_init_fn,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        dataset = self._require_dataset(self._dataset_train, "fit")
        return self._loader(dataset, self._resolve_batch_size(), train=True)

    def val_dataloader(self) -> DataLoader[Any]:
        dataset = self._require_dataset(self._dataset_val, "validate")
        return self._loader(dataset, self._resolve_eval_batch_size(), train=False)

    def test_dataloader(self) -> DataLoader[Any]:
        dataset = self._require_dataset(self._dataset_test, "test")
        return self._loader(dataset, self._resolve_eval_batch_size(), train=False)
