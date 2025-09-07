from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

import numpy as np
import torch
import torch.nn.functional as F
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.models import (
    DynamicModule,
    MultiHeadClassifier,
    MultiTaskModule,
)
from torch import nn

from .lstm_encoder import _LSTM_Encoder
from .multihead_regressor import MultiHeadRegressor
from .seq_model import SequenceModel


class LinearAdapter(nn.Module):
    """Linear adapter for Progressive Neural Networks."""

    def __init__(self, in_features, out_features_per_column, num_prev_modules):
        """
        :param in_features: size of each input sample
        :param out_features_per_column: size of each output sample
        :param num_prev_modules: number of previous modules
        """
        super().__init__()
        # Eq. 1 - lateral connections
        # one layer for each previous column. Empty for the first task.
        self.lat_layers = nn.ModuleList([])
        for _ in range(num_prev_modules):
            m = nn.Linear(in_features, out_features_per_column)
            self.lat_layers.append(m)
        self.num_prev_modules = num_prev_modules

    def forward(self, x):
        if self.num_prev_modules == 0:
            return 0
        assert len(x) == self.num_prev_modules
        hs: List[torch.Tensor] = []
        for ii, lat in enumerate(self.lat_layers):
            el = x[ii]
            if not isinstance(el, torch.Tensor):
                el = torch.as_tensor(el)
            assert el.dim() == 2, (
                "Inputs to LinearAdapter should have two dimensions: "
                "<batch_size, num_features>."
            )
            el = el.to(dtype=lat.weight.dtype, device=lat.weight.device)
            hs.append(lat(el))
        return sum(hs)


class MLPAdapter(nn.Module):
    """MLP adapter for Progressive Neural Networks."""

    def __init__(
        self, in_features, out_features_per_column, num_prev_modules, activation=F.relu
    ):
        """
        :param in_features: size of each input sample
        :param out_features_per_column: size of each output sample
        :param num_prev_modules: number of previous modules
        :param activation: activation function (default=ReLU)
        """
        super().__init__()
        self.num_prev_modules = num_prev_modules
        self.activation = activation

        if num_prev_modules == 0:
            return  # first adapter is empty

        # Eq. 2 - MLP adapter. Not needed for the first task.
        self.V = nn.Linear(in_features * num_prev_modules, out_features_per_column)
        self.alphas = nn.Parameter(torch.randn(num_prev_modules))
        self.U = nn.Linear(out_features_per_column, out_features_per_column)

        self.V.requires_grad = True
        self.alphas.requires_grad = True
        self.U.requires_grad = True

    def forward(self, x):
        if self.num_prev_modules == 0:
            return 0  # first adapter is empty
        assert len(x) == self.num_prev_modules
        assert len(x[0].shape) == 2, (
            "Inputs to MLPAdapter should have two dimensions: "
            "<batch_size, num_features>."
        )
        scaled: List[torch.Tensor] = []
        for i, el in enumerate(x):
            if not isinstance(el, torch.Tensor):
                el = torch.as_tensor(
                    el, dtype=self.alphas.dtype, device=self.alphas.device
                )
            else:
                el = el.to(dtype=self.alphas.dtype, device=self.alphas.device)
            scaled.append(self.alphas[i] * el)
        x = torch.cat(scaled, dim=1)
        x = self.U(self.activation(self.V(x)))
        return x


class PNNColumn(nn.Module):
    """Progressive Neural Network column."""

    def __init__(
        self,
        in_features,
        out_features_per_column,
        num_prev_modules,
        adapter="mlp",
        base_predictor_args=None,
    ):
        """
        :param in_features: size of each input sample
        :param out_features_per_column:
            size of each output sample (single column)
        :param num_prev_modules: number of previous columns
        :param adapter: adapter type. One of {'linear', 'mlp'} or None (default='mlp')
        """
        super().__init__()
        self.in_features = in_features
        self.out_features_per_column = out_features_per_column
        self.num_prev_modules = num_prev_modules

        if self.in_features != base_predictor_args["hidden_sizes"][-1]:
            base_predictor_args["washout"] = 0
        else:
            base_predictor_args["washout"] = 249
        self.itoh = _LSTM_Encoder(input_size=self.in_features, **base_predictor_args)
        if adapter == "linear":
            self.adapter = LinearAdapter(
                in_features, out_features_per_column, num_prev_modules
            )
        elif adapter == "mlp":
            self.adapter = MLPAdapter(
                out_features_per_column, out_features_per_column, num_prev_modules
            )
        elif adapter is None:
            self.adapter = None
        else:
            raise ValueError("`adapter` must be one of: {'mlp', 'linear'} or None.")

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        prev_xs, last_x = x[:-1], x[-1]
        base = self.itoh(last_x)

        if self.adapter is not None and len(prev_xs) > 0:
            # Adapter expects 2-D tensors (batch, features). Remove any singleton
            # sequence dimension from lateral inputs when present.
            prev_xs = np.array(
                [
                    px.squeeze(0) if px.dim() == 3 and px.size(0) == 1 else px
                    for px in prev_xs
                ]
            )
            prev_xs = prev_xs[:, 0]
            assert (
                len(prev_xs) == self.num_prev_modules
            ), f"Expected {self.num_prev_modules} prev modules, got {len(prev_xs)}"

            hs = self.adapter(prev_xs)

            # Ensure shapes match before summation. The encoder returns
            # (seq_len, batch, features); squeeze if the time dimension is singleton.

            hs = hs + base
        else:
            hs = base

        return hs


class PNNLayer(MultiTaskModule):
    """Progressive Neural Network layer.

    The adaptation phase assumes that each experience is a separate task. Multiple
    experiences with the same task label or multiple task labels within the same
    experience will result in a runtime error.
    """

    def __init__(
        self,
        in_features,
        out_features_per_column,
        adapter="mlp",
        base_predictor_args=None,
        use_adapter=True,
    ):
        """
        :param in_features: size of each input sample
        :param out_features_per_column:
            size of each output sample (single column)
        :param adapter: adapter type. One of {'linear', 'mlp'} (default='mlp')
        :param use_adapter: disable lateral adapters when False
        """
        super().__init__()
        self.in_features = in_features
        self.out_features_per_column = out_features_per_column
        self.adapter = adapter
        self.base_predictor_args = base_predictor_args
        self.use_adapter = use_adapter
        # convert from task label to module list order
        self.task_to_module_idx = {}
        first_col = PNNColumn(
            in_features,
            out_features_per_column,
            num_prev_modules=0,
            adapter=adapter if use_adapter else None,
            base_predictor_args=base_predictor_args,
        )
        self.columns = nn.ModuleList([first_col])

    @property
    def num_columns(self):
        return len(self.columns)

    def adaptation(self, experience):
        # Evita MultiTaskModule.adaptation (usa classes...). Prendi solo device ecc.
        DynamicModule.adaptation(self, experience)
        # Usa la tua logica per aggiungere la colonna del NUOVO task
        self.train_adaptation(experience.dataset)

    def train_adaptation(self, dataset: AvalancheDataset):
        # ⚠️ RIMUOVI questa riga che causava l’errore:
        # super().train_adaptation(dataset)
        task_labels_raw = dataset.targets_task_labels
        if isinstance(task_labels_raw, torch.Tensor):
            uniq = torch.unique(task_labels_raw)
            assert len(uniq) == 1, "PNN assumes a single task per experience."
            task_label = int(uniq.item())
        else:
            s = set(task_labels_raw)
            assert len(s) == 1, "PNN assumes a single task per experience."
            task_label = next(iter(s))

        if not self.task_to_module_idx:
            self.task_to_module_idx[task_label] = 0
        else:
            if task_label in self.task_to_module_idx:
                return
            if len(self.task_to_module_idx) == 0:
                self.task_to_module_idx[task_label] = (
                    0  # prima colonna già creata nel __init__
                )
                return
            self.task_to_module_idx[task_label] = self.num_columns
            self._add_column()

    def _add_column(self):
        for p in self.parameters():
            p.requires_grad = False
        self.columns.append(
            PNNColumn(
                self.in_features,
                self.out_features_per_column,
                self.num_columns,
                adapter=self.adapter if self.use_adapter else None,
                base_predictor_args=self.base_predictor_args,
            )
        )

    def forward_single_task(self, x, task_label):
        """Forward.

        :param x: list of inputs.
        :param task_label:
        :return:
        """
        col_idx = self.task_to_module_idx[task_label]
        hs = []
        # print(self.columns)

        for ii in range(col_idx + 1):
            # print(f"x: {np.array(x[: ii + 1]).shape}")

            hs.append(self.columns[ii](x[: ii + 1]))
            # print(f"hs: {hs if len(hs) > 1 else len(hs)}")

        return hs


class PNN(MultiTaskModule):
    """Progressive Neural Network.

    The model assumes that each experience is a separate task. Multiple experiences with
    the same task label or multiple task labels within the same experience will result
    in a runtime error.
    """

    def __init__(
        self,
        num_layers=1,
        in_features=784,
        hidden_features_per_column=100,
        adapter="mlp",
        base_predictor_args=None,
        use_adapter=True,
    ):
        """
        :param num_layers: number of layers (default=1)
        :param in_features: size of each input sample
        :param hidden_features_per_column:
            number of hidden units for each column
        :param adapter: adapter type. One of {'linear', 'mlp'} (default='mlp')
        :param use_adapter: disable lateral adapters when False
        """
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.in_features = in_features
        self.out_features_per_columns = hidden_features_per_column

        self.layers = nn.ModuleList()
        self.layers.append(
            PNNLayer(
                in_features,
                hidden_features_per_column,
                adapter=adapter,
                base_predictor_args=base_predictor_args,
                use_adapter=False,
            )
        )
        for _ in range(num_layers - 1):
            lay = PNNLayer(
                hidden_features_per_column,
                hidden_features_per_column,
                adapter=adapter,
                base_predictor_args=base_predictor_args,
                use_adapter=use_adapter,
            )
            self.layers.append(lay)
        self.regressor = MultiHeadRegressor(
            in_features=hidden_features_per_column, out_features=1
        )

    def adaptation(self, experience):
        # NON chiamare super().adaptation (usa classes_in_this_experience)
        # Aggiorna i layer (aggiunge colonna per il nuovo task)
        for lay in self.layers:
            lay.train_adaptation(experience.dataset)
        # Crea la head del task nella regressor head
        self.regressor.adaptation(experience)

    def forward_single_task(self, x, task_label):
        # x = x.contiguous().view(x.size(0), self.in_features)

        num_columns = self.layers[0].num_columns
        col_idx = self.layers[-1].task_to_module_idx[task_label]

        x = [x for _ in range(num_columns)]
        for lay in self.layers:
            x = [F.relu(el) for el in lay.forward_single_task(x, task_label)]
        # ⬇️ usa la regressor head
        return self.regressor.forward_single_task(x[col_idx], task_label)


# Backward compatibility
LSTMPNN = PNN
