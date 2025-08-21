"""Progressive Neural Network with LSTM columns."""

from __future__ import annotations

from typing import Literal

import torch
from avalanche.models import PNN
from avalanche.models.pnn import (
    MultiHeadClassifier,
    PNNLayer,
)
from torch import nn

from .lstm_encoder import LSTMEncoder


class _LSTMColumn(nn.Module):
    """PNN column composed of an LSTM encoder and lateral adapters."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_prev_modules: int,
        adapter: Literal["linear", "mlp"] = "mlp",
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        from avalanche.models.pnn import (
            LinearAdapter,
            MLPAdapter,
        )

        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        if adapter == "linear":
            self.adapter = LinearAdapter(hidden_size, hidden_size, num_prev_modules)
        else:
            self.adapter = MLPAdapter(hidden_size, hidden_size, num_prev_modules)

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        prev_xs, last_x = xs[:-1], xs[-1]
        hs = self.adapter(prev_xs) if prev_xs else 0
        hs = hs + self.encoder(last_x)
        return hs


class _LSTMPNNLayer(PNNLayer):
    """PNN layer using :class:`_LSTMColumn` columns."""

    def __init__(
        self,
        in_features: int,
        out_features_per_column: int,
        *,
        adapter: Literal["linear", "mlp"] = "mlp",
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(in_features, out_features_per_column, adapter=adapter)
        self.num_layers = num_layers
        self.dropout = dropout
        self.columns = nn.ModuleList(
            [
                _LSTMColumn(
                    in_features,
                    out_features_per_column,
                    0,
                    adapter=adapter,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            ]
        )

    def _add_column(self) -> None:  # type: ignore[override]
        for param in self.parameters():
            param.requires_grad = False
        self.columns.append(
            _LSTMColumn(
                self.in_features,
                self.out_features_per_column,
                self.num_columns,
                adapter=self.adapter,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
        )


class LSTMPNN(PNN):
    """Progressive Neural Network using LSTM encoders and regression heads."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
        adapter: Literal["linear", "mlp"] = "mlp",
    ) -> None:
        super().__init__(
            num_layers=1,
            in_features=input_size,
            hidden_features_per_column=hidden_size,
            adapter=adapter,
        )
        # override default layer with LSTM-based one
        self.layers = nn.ModuleList(
            [
                _LSTMPNNLayer(
                    input_size,
                    hidden_size,
                    adapter=adapter,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            ]
        )
        # regression head with one output per task
        self.classifier = MultiHeadClassifier(
            hidden_size, initial_out_features=1, masking=False
        )

    def forward_single_task(self, x: torch.Tensor, task_label: int):  # type: ignore[override]
        num_columns = self.layers[0].num_columns
        xs = [x for _ in range(num_columns)]
        for lay in self.layers:
            xs = [torch.relu(el) for el in lay(xs, task_label)]
        col_idx = self.layers[-1].task_to_module_idx[task_label]
        return self.classifier(xs[col_idx], task_label)


__all__ = ["LSTMPNN"]
