import torch
from types import SimpleNamespace

from spaceai.models.predictors.multihead_regressor import MultiHeadRegressor


class _DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples: int, in_features: int, task_label: int):
        self.x = torch.randn(n_samples, in_features)
        self.y = torch.randn(n_samples, 1)
        self.targets_task_labels = torch.full((n_samples,), task_label)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def test_adaptation_adds_new_head():
    reg = MultiHeadRegressor(in_features=3, out_features=1)

    ds0 = _DummyDataset(5, 3, task_label=0)
    exp0 = SimpleNamespace(dataset=ds0)
    reg.adaptation(exp0)
    assert "0" in reg.heads

    ds1 = _DummyDataset(5, 3, task_label=1)
    exp1 = SimpleNamespace(dataset=ds1)
    reg.adaptation(exp1)
    assert "1" in reg.heads

    out = reg(torch.randn(2, 3), torch.tensor([1]))
    assert out.shape == (2, 1)
