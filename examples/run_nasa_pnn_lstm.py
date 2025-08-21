"""Run NASA PNN-LSTM experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from avalanche.training import Naive
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from spaceai.benchmarks.nasa_benchmark import make_nasa_benchmark
from spaceai.eval.dynamic_threshold import detect_anomalies
from spaceai.eval.range_metrics import range_metrics
from spaceai.models.pnn_lstm import LSTMPNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--dataset", choices=["smap", "msl"], required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    train_stream, test_stream, meta = make_nasa_benchmark(
        args.root, args.dataset, seq_len=args.seq_len, stride=args.stride
    )

    model = LSTMPNN(
        input_size=1,
        hidden_size=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    strategy = Naive(
        model,
        optimizer,
        criterion,
        train_mb_size=args.batch_size,
        eval_mb_size=args.batch_size,
        device="cpu",
    )

    out_dir = Path("outputs") / "nasa_pnn_lstm"
    out_dir.mkdir(parents=True, exist_ok=True)

    for exp_id, (train_exp, test_exp) in enumerate(zip(train_stream, test_stream)):
        model.adaptation(train_exp)
        strategy.train(train_exp, max_epochs=args.epochs)

        preds = []
        targets = []
        for x, y, t, _ in DataLoader(
            test_exp.dataset, batch_size=args.batch_size, shuffle=False
        ):
            pred = model(x, t)
            preds.append(pred.detach().cpu())
            targets.append(y.detach().cpu())
        preds_t = torch.cat(preds).squeeze()
        targets_t = torch.cat(targets).squeeze()
        errors = (preds_t - targets_t).numpy() ** 2

        intervals = detect_anomalies(errors)
        metrics = range_metrics(intervals, [])

        cid = meta["mapping"][exp_id]
        np.savetxt(
            out_dir / f"{cid}_anomalies.csv",
            np.array(intervals),
            fmt="%d",
            delimiter=",",
        )
        with open(out_dir / f"{cid}_metrics.json", "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    main()
