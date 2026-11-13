"""Training sampler that keeps every positive tile and re-samples the negative ones each epoch."""

import math
import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class EpochNegativeSampler(Sampler[int]):
    """Each epoch: all positive tiles + ``round(negative_ratio * negatives)`` negatives drawn with ``seed:epoch``.

    The epoch length is fixed, at least ``min_length`` and a multiple of ``batch_unit`` (``batch_size *
    grad_accum_steps``), padding with random repeats of the epoch's own samples, so ``drop_last`` never drops data
    and the optimizer never steps on a partial accumulation window (same goal as RF-DETR's
    ``GradAccumAlignedDataset``). Lightning calls :meth:`set_epoch` before every epoch.
    """

    def __init__(
        self, positive: Sequence[bool], negative_ratio: float, *, batch_unit: int = 1, min_length: int = 0,
        seed: int = 0,
    ) -> None:
        if not 0 <= negative_ratio <= 1:
            raise ValueError("negative_ratio debe estar en [0, 1].")
        if batch_unit < 1:
            raise ValueError("batch_unit debe ser >= 1.")
        self.positives = [index for index, flag in enumerate(positive) if flag]
        self.negatives = [index for index, flag in enumerate(positive) if not flag]
        self.negatives_per_epoch = round(negative_ratio * len(self.negatives))
        self.base_length = len(self.positives) + self.negatives_per_epoch
        if self.base_length == 0:
            raise ValueError("No hay tiles de entrenamiento (ni positivos ni negativos muestreados).")
        target = max(self.base_length, min_length)
        self.length = math.ceil(target / batch_unit) * batch_unit
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(f"{self.seed}:{self.epoch}")
        indices = self.positives + rng.sample(self.negatives, self.negatives_per_epoch)
        indices += rng.choices(indices, k=self.length - len(indices))
        rng.shuffle(indices)
        return iter(indices)
