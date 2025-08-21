"""LSTM encoder module."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class LSTMEncoder(nn.Module):
    """Simple LSTM encoder returning the last hidden state.

    Parameters
    ----------
    input_size:
        Number of input features for each time step.
    hidden_size:
        Number of features in the hidden state.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability applied between LSTM layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input sequence and return the last hidden state."""
        output, _ = self.lstm(x)
        return output[:, -1, :]
