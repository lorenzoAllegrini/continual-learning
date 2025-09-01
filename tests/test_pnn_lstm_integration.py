from types import SimpleNamespace

import numpy as np
import pytest
import torch

try:
    from spaceai.benchmarks.nasa_benchmark import make_nasa_benchmark
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    make_nasa_benchmark = None
from spaceai.models.predictors.pnn_lstm import LSTMPNN


def _build_fake_dataset(root):
    for split in ["train", "test"]:
        d = root / "smap" / split
        d.mkdir(parents=True, exist_ok=True)
        for ch in ["A", "B"]:
            np.save(d / f"{ch}.npy", np.linspace(0, 1, 200, dtype=np.float32))


@pytest.mark.skipif(make_nasa_benchmark is None, reason="NASA benchmark not available")
def test_benchmark_tasks(tmp_path):
    _build_fake_dataset(tmp_path)
    train_stream, test_stream, meta = make_nasa_benchmark(
        str(tmp_path), "smap", seq_len=10, stride=1
    )
    assert len(train_stream) == 2
    assert len(test_stream) == 2
    assert meta["n_tasks"] == 2


def test_lstm_pnn_forward():
    model = LSTMPNN(
        in_features=1,
        hidden_features_per_column=8,
        base_predictor_args={"hidden_sizes": [8], "dropout": 0.0},
    )
    exp = SimpleNamespace(dataset=None, task_labels=[0], classes_in_this_experience=[0])
    model.layers[0].task_to_module_idx[0] = 0
    out = model(torch.randn(4, 10, 1), 0)
    assert out.shape == (4, 1)


def test_disable_adapters():
    model = LSTMPNN(
        in_features=1,
        hidden_features_per_column=8,
        num_layers=2,
        use_adapter=False,
        base_predictor_args={"hidden_sizes": [8], "dropout": 0.0},
    )
    for lay in model.layers:
        for col in lay.columns:
            assert col.adapter is None


@pytest.mark.skipif(make_nasa_benchmark is None, reason="NASA benchmark not available")
def test_training_step(tmp_path):
    _build_fake_dataset(tmp_path)
    train_stream, _, _ = make_nasa_benchmark(
        str(tmp_path), "smap", seq_len=10, stride=1
    )
    exp = train_stream[0]
    model = LSTMPNN(
        in_features=1,
        hidden_features_per_column=8,
        base_predictor_args={"hidden_sizes": [8], "dropout": 0.0},
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    model.layers[0].task_to_module_idx[0] = 0
    loader = torch.utils.data.DataLoader(exp.dataset, batch_size=8)
    x, y, t, _ = next(iter(loader))
    y = torch.stack(y).float()
    before = model.layers[0].columns[0].encoder.lstm.weight_hh_l0.clone()
    optimizer.zero_grad()
    out = model(x, 0).squeeze()
    loss = criterion(out, y.squeeze())
    loss.backward()
    optimizer.step()
    after = model.layers[0].columns[0].encoder.lstm.weight_hh_l0
    assert not torch.equal(before, after)
