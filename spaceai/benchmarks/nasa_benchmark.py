"""Benchmark for NASA SMAP & MSL datasets."""

from __future__ import annotations

from random import Random
from types import SimpleNamespace
from typing import (
    Dict,
    Literal,
    Tuple,
)

from spaceai.data.nasa_smap_msl import (
    load_nasa,
    make_nasa_datasets,
)


def make_nasa_benchmark(
    root: str,
    dataset: Literal["smap", "msl"],
    seq_len: int = 128,
    stride: int = 1,
    order_seed: int = 0,
) -> Tuple[list[SimpleNamespace], list[SimpleNamespace], Dict[str, object]]:
    """Create continual learning benchmark for NASA datasets."""

    channels = list(load_nasa(root, dataset, "train").keys())
    rng = Random(order_seed)
    rng.shuffle(channels)

    train_ds, test_ds, mapping = make_nasa_datasets(
        root,
        dataset,
        seq_len=seq_len,
        stride=stride,
        order=channels,
    )
    train_stream = [
        SimpleNamespace(
            dataset=ds,
            task_labels=[i],
            classes_in_this_experience=[0],
            current_experience=i,
        )
        for i, ds in enumerate(train_ds)
    ]
    test_stream = [
        SimpleNamespace(
            dataset=ds,
            task_labels=[i],
            classes_in_this_experience=[0],
            current_experience=i,
        )
        for i, ds in enumerate(test_ds)
    ]
    meta: Dict[str, object] = {
        "mapping": mapping,
        "in_dim": 1,
        "n_tasks": len(channels),
    }
    return train_stream, test_stream, meta


__all__ = ["make_nasa_benchmark"]
