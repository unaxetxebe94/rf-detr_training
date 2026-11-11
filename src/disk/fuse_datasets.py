"""Compatibility entry point; fusion now belongs to preprocessing."""

import sys

from src.preprocess.fusion import (
    _discover_preprocessed_datasets,
    _fuse_datasets,
    _fuse_splitted_datasets,
    _source_sort_key,
    main,
)

__all__ = [
    "_discover_preprocessed_datasets",
    "_fuse_datasets",
    "_fuse_splitted_datasets",
    "_source_sort_key",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
