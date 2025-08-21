"""Dynamic thresholding for anomaly detection."""

from __future__ import annotations

from typing import (
    List,
    Tuple,
)

import numpy as np


def detect_anomalies(
    errors: np.ndarray,
    *,
    smooth: str = "ewma",
    alpha: float = 0.3,
    method: str = "quantile",
    q: float = 0.997,
    min_len: int = 5,
    min_gap: int = 3,
) -> List[Tuple[int, int]]:
    """Detect anomalous ranges from reconstruction errors."""

    series = errors.astype(float)
    if smooth == "ewma":
        smoothed = np.copy(series)
        for i in range(1, len(smoothed)):
            smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * smoothed[i]
    else:
        smoothed = series

    if method == "quantile":
        thr = np.quantile(smoothed, q)
    else:
        thr = smoothed.mean() + q * smoothed.std()

    mask = smoothed > thr
    intervals: List[Tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_len:
                intervals.append((start, idx))
            start = None
    if start is not None and len(mask) - start >= min_len:
        intervals.append((start, len(mask)))

    # merge close intervals
    merged: List[Tuple[int, int]] = []
    for s, e in intervals:
        if merged and s - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged
