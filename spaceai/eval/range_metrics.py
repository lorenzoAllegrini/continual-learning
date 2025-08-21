"""Range-based evaluation metrics."""

from __future__ import annotations

from typing import (
    Iterable,
    List,
    Tuple,
)

import numpy as np

Range = Tuple[int, int]


def _intersection(a: Range, b: Range) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def range_metrics(predicted: Iterable[Range], ground_truth: Iterable[Range]) -> dict:
    """Compute precision, recall, F1 and latency for ranges."""

    pred = list(predicted)
    gt = list(ground_truth)
    pred_len = sum(e - s for s, e in pred)
    gt_len = sum(e - s for s, e in gt)
    overlap = sum(_intersection(p, g) for p in pred for g in gt)
    precision = overlap / pred_len if pred_len > 0 else 0.0
    recall = overlap / gt_len if gt_len > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    latencies: List[float] = []
    for g in gt:
        matches = [p for p in pred if _intersection(p, g) > 0]
        if matches:
            first = min(matches, key=lambda r: r[0])
            latencies.append(max(0, first[0] - g[0]))
    latency = float(np.mean(latencies)) if latencies else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency": latency,
    }
