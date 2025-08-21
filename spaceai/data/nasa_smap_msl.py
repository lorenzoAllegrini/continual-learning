"""Utilities for NASA SMAP & MSL datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Literal,
)

import numpy as np
import torch
from avalanche.benchmarks.utils import (
    data_attribute,
    make_avalanche_dataset,
)
from torch.utils.data import TensorDataset


def load_nasa(
    root: str, dataset: Literal["smap", "msl"], split: Literal["train", "test"]
) -> Dict[str, np.ndarray]:
    """Load NASA dataset split.

    The function expects files organized as ``root/dataset/split/*.npy`` where
    each file corresponds to a channel.
    """

    base = Path(root) / dataset / split
    data: Dict[str, np.ndarray] = {}
    for file in base.glob("*.npy"):
        arr = np.load(file)
        data[file.stem] = arr.astype(np.float32)
    return data


def _window_array(
    arr: np.ndarray, seq_len: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for start in range(0, len(arr) - seq_len, stride):
        xs.append(arr[start : start + seq_len])
        ys.append(arr[start + seq_len])
    x = np.stack(xs) if xs else np.empty((0, seq_len))
    y = np.stack(ys) if ys else np.empty((0,))
    return x[..., None], y[..., None]


def _make_dataset(
    series: np.ndarray,
    channel_idx: int,
    mean: float,
    std: float,
    seq_len: int,
    stride: int,
):
    norm = (series - mean) / (std if std > 0 else 1.0)
    x, y = _window_array(norm, seq_len, stride)
    patterns = torch.from_numpy(x).float()
    base = TensorDataset(patterns)
    targets_attr = data_attribute.DataAttribute(
        torch.from_numpy(y).float(), "targets", use_in_getitem=True
    )
    task_attr = data_attribute.DataAttribute(
        torch.full((len(y),), channel_idx, dtype=torch.long),
        "targets_task_labels",
        use_in_getitem=True,
    )
    chan_attr = data_attribute.DataAttribute(
        torch.full((len(y),), channel_idx, dtype=torch.long),
        "channel_id",
        use_in_getitem=True,
    )
    return make_avalanche_dataset(
        base, data_attributes=[targets_attr, task_attr, chan_attr]
    )


def make_nasa_datasets(
    root: str,
    dataset: Literal["smap", "msl"],
    *,
    seq_len: int,
    stride: int,
    order: Iterable[str],
) -> tuple[list[TensorDataset], list[TensorDataset], Dict[int, str]]:
    train_raw = load_nasa(root, dataset, "train")
    test_raw = load_nasa(root, dataset, "test")
    mapping: Dict[int, str] = {}
    train_datasets: list[TensorDataset] = []
    test_datasets: list[TensorDataset] = []
    for idx, cid in enumerate(order):
        train_series = train_raw[cid]
        test_series = test_raw[cid]
        mean = float(train_series.mean())
        std = float(train_series.std())
        train_datasets.append(
            _make_dataset(train_series, idx, mean, std, seq_len, stride)
        )
        test_datasets.append(
            _make_dataset(test_series, idx, mean, std, seq_len, stride)
        )
        mapping[idx] = cid
    return train_datasets, test_datasets, mapping
