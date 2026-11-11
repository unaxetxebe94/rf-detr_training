"""Batch sampler that keeps every original image on a single DataLoader worker."""

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class OriginAffineBatchSampler(Sampler[list[int]]):
    """Groups samples by original image so each worker decodes each original once per epoch.

    Originals are distributed (balanced by number of samples) over `num_streams` streams, one per
    DataLoader worker. A stream consumes its originals in windows of `images_per_stream`; with
    `shuffle` the samples of a window are mixed, so a batch can contain tiles of several originals
    while the worker only holds that window in memory. Batches are emitted round-robin over the
    streams, which matches the round-robin order in which the DataLoader hands batches to its
    workers (batch b -> worker b % num_workers).

    With `pad` (training), every stream is completed with repeated samples of its last window up to
    the same number of batches, rounded so that the total is a multiple of `batch_multiple`
    (gradient accumulation) and no optimizer step uses a partial window. Without `pad`
    (evaluation), every sample is emitted exactly once and the last batch of a stream may be short.
    """

    def __init__(
        self,
        sample_origins: Sequence[int],
        batch_size: int,
        *,
        num_streams: int = 1,
        images_per_stream: int = 1,
        shuffle: bool = False,
        pad: bool = False,
        batch_multiple: int = 1,
        seed: int = 0,
    ) -> None:
        if batch_size < 1 or num_streams < 1 or images_per_stream < 1 or batch_multiple < 1:
            raise ValueError("batch_size, num_streams, images_per_stream y batch_multiple deben ser >= 1.")
        self.batch_size = batch_size
        self.num_streams = num_streams
        self.images_per_stream = images_per_stream
        self.shuffle = shuffle
        self.pad = pad
        self.batch_multiple = batch_multiple
        self.seed = seed
        self.epoch = 0

        groups = defaultdict(list)
        for index, origin in enumerate(sample_origins):
            groups[origin].append(index)
        self.groups = dict(groups)
        # Fixed assignment (largest first to the least loaded stream) keeps len() constant across epochs.
        self.streams = [[] for _ in range(num_streams)]
        loads = [0] * num_streams
        for origin in sorted(self.groups, key=lambda key: (-len(self.groups[key]), key)):
            stream = min(range(num_streams), key=lambda s: (loads[s], s))
            self.streams[stream].append(origin)
            loads[stream] += len(self.groups[origin])
        self.stream_batches = [math.ceil(load / batch_size) for load in loads]
        self.padded_batches = 0
        if pad and any(self.stream_batches):
            # Empty streams (more workers than originals) cannot be padded with their own samples.
            active = sum(1 for count in self.stream_batches if count)
            per_stream = max(self.stream_batches)
            while (per_stream * active) % batch_multiple:
                per_stream += 1
            self.padded_batches = per_stream

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        if self.padded_batches:
            return self.padded_batches * sum(1 for count in self.stream_batches if count)
        return sum(self.stream_batches)

    def _stream_batches(self, origins: list[int], rng: random.Random) -> list[list[int]]:
        origins = list(origins)
        if self.shuffle:
            rng.shuffle(origins)
        sequence, last_window = [], []
        for start in range(0, len(origins), self.images_per_stream):
            window = [index for origin in origins[start:start + self.images_per_stream] for index in self.groups[origin]]
            if self.shuffle:
                rng.shuffle(window)
            sequence.extend(window)
            last_window = window
        if self.pad and sequence:
            target = self.padded_batches * self.batch_size
            sequence.extend(rng.choice(last_window) for _ in range(target - len(sequence)))
        return [sequence[start:start + self.batch_size] for start in range(0, len(sequence), self.batch_size)]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(f"{self.seed}:{self.epoch}")
        self.epoch += 1
        streams = [self._stream_batches(origins, rng) for origins in self.streams]
        streams = [batches for batches in streams if batches]
        for position in range(max((len(batches) for batches in streams), default=0)):
            for batches in streams:
                if position < len(batches):
                    yield batches[position]
