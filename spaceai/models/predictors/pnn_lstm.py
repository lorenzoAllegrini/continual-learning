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

    def forward(self, x):
        assert len(x) == self.num_prev_modules
        hs = []
        for ii, lat in enumerate(self.lat_layers):
            hs.append(lat(x[ii]))
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

    def forward(self, x):
        if self.num_prev_modules == 0:
            return 0  # first adapter is empty

        assert len(x) == self.num_prev_modules
        assert len(x[0].shape) == 2, (
            "Inputs to MLPAdapter should have two dimensions: "
            "<batch_size, num_features>."
        )
        for i, el in enumerate(x):
            x[i] = self.alphas[i] * el
        x = torch.cat(x, dim=1)
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
        :param adapter: adapter type. One of {'linear', 'mlp'} (default='mlp')
        """
        super().__init__()
        self.in_features = in_features
        self.out_features_per_column = out_features_per_column
        self.num_prev_modules = num_prev_modules

        self.itoh = _LSTM_Encoder(**base_predictor_args)
        if adapter == "linear":
            self.adapter = LinearAdapter(
                in_features, out_features_per_column, num_prev_modules
            )
        elif adapter == "mlp":
            self.adapter = MLPAdapter(
                out_features_per_column, out_features_per_column, num_prev_modules
            )
        else:
            raise ValueError("`adapter` must be one of: {'mlp', `linear'}.")

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        print(np.array(x).shape)
        prev_xs, last_x = x[:-1], x[-1]
        print(np.array(prev_xs).shape)
        print(prev_xs)
        hs = self.adapter(prev_xs)
        hs += self.itoh(last_x)
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
    ):
        """
        :param in_features: size of each input sample
        :param out_features_per_column:
            size of each output sample (single column)
        :param adapter: adapter type. One of {'linear', 'mlp'} (default='mlp')
        """
        super().__init__()
        self.in_features = in_features
        self.out_features_per_column = out_features_per_column
        self.adapter = adapter
        self.base_predictor_args = base_predictor_args
        # convert from task label to module list order
        self.task_to_module_idx = {}
        first_col = PNNColumn(
            in_features,
            out_features_per_column,
            0,
            adapter=adapter,
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
                adapter=self.adapter,
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
        for ii in range(col_idx + 1):
            print(self.columns)
            hs.append(self.columns[ii](x[: ii + 1]))
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
    ):
        """
        :param num_layers: number of layers (default=1)
        :param in_features: size of each input sample
        :param hidden_features_per_column:
            number of hidden units for each column
        :param adapter: adapter type. One of {'linear', 'mlp'} (default='mlp')
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
            )
        )
        for _ in range(num_layers - 1):
            lay = PNNLayer(
                hidden_features_per_column,
                hidden_features_per_column,
                adapter=adapter,
                base_predictor_args=base_predictor_args,
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
        #x = x.contiguous().view(x.size(0), self.in_features)

        num_columns = self.layers[0].num_columns
        col_idx = self.layers[-1].task_to_module_idx[task_label]

        x = [x for _ in range(num_columns)]
        for lay in self.layers:
            x = [F.relu(el) for el in lay.forward_single_task(x, task_label)]
        # ⬇️ usa la regressor head
        return self.regressor.forward_single_task(x[col_idx], task_label)


class PNNSequenceModel(SequenceModel):
    """Wrapper che rende una PNN compatibile con SequenceModel."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        adapter: str = "mlp",
        device: Literal["cpu", "cuda"] = "cpu",
        stateful: bool = False,
        reduce_out: Optional[Literal["first", "mean"]] = None,
        washout: int = 0,
    ):
        super().__init__(
            device, stateful=stateful, reduce_out=reduce_out, washout=washout
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.adapter = adapter

    def build_fn(self) -> nn.Module:
        """Costruisce la PNN vera e propria."""
        return PNN(
            num_layers=self.num_layers,
            in_features=self.input_size,
            hidden_features_per_column=self.hidden_size,
            adapter=self.adapter,
        )

    def __call__(self, input: torch.Tensor, task_label: int = 0):
        """Forward che gestisce anche i task label."""
        if self.model is None:
            raise ValueError("Model must be built before calling predict.")

        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).float().to(self.device)

        # qui la PNN richiede forward_single_task con task_label
        out = self.model.forward_single_task(input, task_label)

        if self.reduce_out is None:
            return out
        elif self.reduce_out == "mean":
            return out.mean(dim=-1)
        elif self.reduce_out == "first":
            return out[..., 0]
        else:
            raise ValueError(f"Invalid reduce_out value: {self.reduce_out}")
