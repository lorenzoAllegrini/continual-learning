import torch
import torch.nn.functional as F
from torch import nn
from .lstm_encoder import _LSTM_Encoder
from .predictors.seq_model import SequenceModel
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence
from avalanche.models import MultiTaskModule, DynamicModule
from avalanche.models import MultiHeadClassifier

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

class LinearAdapter(nn.Module):
    """
    Linear adapter for Progressive Neural Networks.
    """
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
    """
     MLP adapter for Progressive Neural Networks.
    """
    def __init__(self, in_features, out_features_per_column, num_prev_modules,
                 activation=F.relu):
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
        self.V = nn.Linear(in_features * num_prev_modules,
                           out_features_per_column)
        self.alphas = nn.Parameter(torch.randn(num_prev_modules))
        self.U = nn.Linear(out_features_per_column, out_features_per_column)

    def forward(self, x):
        if self.num_prev_modules == 0:
            return 0  # first adapter is empty

        assert len(x) == self.num_prev_modules
        assert len(x[0].shape) == 2, \
            "Inputs to MLPAdapter should have two dimensions: " \
            "<batch_size, num_features>."
        for i, el in enumerate(x):
            x[i] = self.alphas[i] * el
        x = torch.cat(x, dim=1)
        x = self.U(self.activation(self.V(x)))
        return x


class PNNColumn(nn.Module):
    """
    Progressive Neural Network column.
    """
    def __init__(self, in_features, hidden_size, out_features_per_column, num_prev_modules, dropout,
                 adapter='mlp'):
        """
        :param in_features: size of each input sample
        :param out_features_per_column:
            size of each output sample (single column)
        :param num_prev_modules: number of previous columns
        :param adapter: adapter type. One of {'linear', 'mlp'} (default='mlp')
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_features_per_column = out_features_per_column
        self.num_prev_modules = num_prev_modules

        self.itoh = _LSTM_Encoder(in_features, hidden_sizes=hidden_size, dropout=dropout)
        if adapter == 'linear':
            self.adapter = LinearAdapter(hidden_size, out_features_per_column,
                                         num_prev_modules)
        elif adapter == 'mlp':
            self.adapter = MLPAdapter(hidden_size, out_features_per_column,
                                      num_prev_modules)
        else:
            raise ValueError("`adapter` must be one of: {'mlp', `linear'}.")

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        prev_xs, last_x = x[:-1], x[-1]
        hs = self.adapter(prev_xs)
        hs += self.itoh(last_x)
        return hs


class PNNLayer(MultiTaskModule):
    """ Progressive Neural Network layer.

        The adaptation phase assumes that each experience is a separate task.
        Multiple experiences with the same task label or multiple task labels
        within the same experience will result in a runtime error.
        """
    def __init__(self, in_features, out_features_per_column, hidden_size, adapter='mlp'):
        """
        :param in_features: size of each input sample
        :param out_features_per_column:
            size of each output sample (single column)
        :param adapter: adapter type. One of {'linear', 'mlp'} (default='mlp')
        """
        super().__init__()
        self.in_features = in_features
        self.out_features_per_column = out_features_per_column
        self.hidden_size = hidden_size
        self.adapter = adapter

        # convert from task label to module list order
        self.task_to_module_idx = {}
        first_col = PNNColumn(in_features, hidden_size, out_features_per_column,
                              0, adapter=adapter)
        self.columns = nn.ModuleList([first_col])

    @property
    def num_columns(self):
        return len(self.columns)

    def train_adaptation(self, dataset: AvalancheDataset):
        """ Training adaptation for PNN layer.

        Adds an additional column to the layer.

        :param dataset:
        :return:
        """
        super().train_adaptation(dataset)
        task_labels = dataset.targets_task_labels
        if isinstance(task_labels, ConstantSequence):
            # task label is unique. Don't check duplicates.
            task_labels = [task_labels[0]]
        else:
            task_labels = set(task_labels)
        assert len(task_labels) == 1, \
            "PNN assumes a single task for each experience. Please use a " \
            "compatible benchmark."
        # extract task label from set
        task_label = next(iter(task_labels))
        assert task_label not in self.task_to_module_idx, \
            "A new experience is using a previously seen task label. This is " \
            "not compatible with PNN, which assumes different task labels for" \
            " each training experience."

        if len(self.task_to_module_idx) == 0:
            # we have already initialized the first column.
            # No need to call add_column here.
            self.task_to_module_idx[task_label] = 0
        else:
            self.task_to_module_idx[task_label] = self.num_columns
            self._add_column()

    def _add_column(self):
        """ Add a new column. """
        # Freeze old parameters
        for param in self.parameters():
            param.requires_grad = False
        self.columns.append(PNNColumn(self.in_features,
                                      self.out_features_per_column,
                                      self.num_columns,
                                      adapter=self.adapter))

    def forward_single_task(self, x, task_label):
        """ Forward.

        :param x: list of inputs.
        :param task_label:
        :return:
        """
        col_idx = self.task_to_module_idx[task_label]
        hs = []
        for ii in range(col_idx + 1):
            hs.append(self.columns[ii](x[:ii+1]))
        return hs


# ---- La tua SequenceModel come fornita ----
# (incolla qui la definizione della tua SequenceModel o importala)

class PNNCore(nn.Module):
    """
    Core PNN senza MultiTaskModule. Tiene i layers e una head per colonna/task.
    """
    def __init__(self, in_channels: int, hidden: int, num_layers: int = 1,
                 adapter: str = 'mlp', dropout: float = 0.0, out_channels: Optional[int] = None):
        super().__init__()
        assert num_layers >= 1
        self.in_channels = in_channels
        self.hidden = hidden
        self.num_layers = num_layers
        self.adapter = adapter
        self.dropout = dropout
        self.out_channels = out_channels if out_channels is not None else in_channels

        # layers
        self.layers = nn.ModuleList()
        # primo layer usa in_channels come in_features, out_features_per_column = hidden (embedding)
        self.layers.append(PNNLayer(in_channels, hidden, hidden, dropout=dropout, adapter=adapter))
        for _ in range(num_layers - 1):
            # layers successivi: in_features=hidden, out_features_per_column=hidden
            self.layers.append(PNNLayer(hidden, hidden, hidden, dropout=dropout, adapter=adapter))

        # heads (una per colonna/task): hidden -> out_channels
        self.heads = nn.ModuleList([nn.Linear(hidden, self.out_channels)])

        # gestione task/colonne
        self.task_to_idx: Dict[Union[int, str], int] = {}
        self.idx_to_task: List[Union[int, str]] = []
        # primo task
        self.task_to_idx[0] = 0
        self.idx_to_task.append(0)
        self.active_task_idx: int = 0

    @property
    def num_columns(self):
        return self.layers[0].num_columns

    def set_active_task(self, task_label: Union[int, str]):
        if task_label not in self.task_to_idx:
            raise ValueError(f"Task {task_label} non esistente. Aggiungilo con add_task().")
        self.active_task_idx = self.task_to_idx[task_label]

    def add_task(self, task_label: Union[int, str]):
        if task_label in self.task_to_idx:
            raise ValueError(f"Task {task_label} già presente.")
        # aggiungi una colonna a ogni layer
        for lay in self.layers:
            lay.add_column()
        # nuova head
        self.heads.append(nn.Linear(self.hidden, self.out_channels))
        new_idx = self.num_columns - 1
        self.task_to_idx[task_label] = new_idx
        self.idx_to_task.append(task_label)
        # setta active su questo nuovo task
        self.active_task_idx = new_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C) -> pred (B, out_channels) per il task attivo
        """
        # costruisci x_bases per il primo layer (stessa sequenza ripetuta N volte)
        x_bases = [x for _ in range(self.num_columns)]
        # passa layer per layer
        for li, lay in enumerate(self.layers):
            hs = lay.forward_all_columns(x_bases)   # lista di (B, H)
            # per il layer successivo, l'input base per colonna è... (B, L?, C?) — ma noi usiamo embeddings
            # quindi costruiamo sequenze "degenerate": qui usiamo direttamente l'embedding come base per il prossimo layer.
            # Alternativa: far sì che i layer successivi usino in_features=hidden e che PNNColumn si aspetti (B, H) già pronti.
            # Adattiamo PNNColumn per accettare come "base" un embedding (B, H) con un wrapper fittizio:
            x_bases = [h.unsqueeze(1) for h in hs]  # (B, 1, H) - così _LSTM_Encoder accetta "sequenza" di len=1
        # head del task attivo
        col = self.active_task_idx
        y_hat = self.heads[col](hs[col])  # (B, out_channels)
        return y_hat


class PNN(SequenceModel):
    def __init__(
        self,
        in_channels: int,
        hidden: int,
        num_layers: int = 1,
        adapter: str = "mlp",
        dropout: float = 0.0,
        out_channels: Optional[int] = None,
        device: Literal["cpu", "cuda"] = "cpu",
        stateful: bool = False,           # non usato qui, ma compatibile
        reduce_out: Optional[Literal["first", "mean"]] = None,
        washout: int = 0,
    ):
        super().__init__(device=device, stateful=stateful, reduce_out=reduce_out, washout=washout)
        self.cfg = dict(
            in_channels=in_channels,
            hidden=hidden,
            num_layers=num_layers,
            adapter=adapter,
            dropout=dropout,
            out_channels=out_channels
        )
        self.build()  # crea self.model

    # SequenceModel richiede questo
    def build_fn(self) -> torch.nn.Module:
        core = PNNCore(**self.cfg)
        return core

    # API comode che inoltrano al core
    def add_task(self, task_label: Union[int, str]):
        core: PNNCore = self.model  # type: ignore
        core.add_task(task_label)

    def set_active_task(self, task_label: Union[int, str]):
        core: PNNCore = self.model  # type: ignore
        core.set_active_task(task_label)

