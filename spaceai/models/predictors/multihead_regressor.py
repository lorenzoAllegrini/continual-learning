import torch
from torch import nn
from avalanche.models.dynamic_modules import DynamicModule

class MultiHeadRegressor(DynamicModule):
    """
    Multi-head regressor: una testa lineare per ogni task label.
    Non richiede classes_in_this_experience né masking.
    """
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = nn.ModuleDict()

    def _get_single_task_label(self, ds):
        tl = ds.targets_task_labels
        if isinstance(tl, torch.Tensor):
            uniq = torch.unique(tl)
            assert len(uniq) == 1, "Regressor: ogni esperienza deve avere un solo task label."
            return int(uniq.item())
        else:
            s = set(tl)
            assert len(s) == 1, "Regressor: ogni esperienza deve avere un solo task label."
            return int(next(iter(s)))

    def forward(self, x: torch.Tensor, task_labels) -> torch.Tensor:
        # batch single-task (tipico con Avalanche) → accetta int o tensore costante
        if isinstance(task_labels, int):
            return self.forward_single_task(x, task_labels)
        if isinstance(task_labels, torch.Tensor):
            uniq = torch.unique(task_labels)
            assert len(uniq) == 1, "Batch multi-task non supportato in questa head."
            return self.forward_single_task(x, int(uniq.item()))
        # lista/iterabile
        s = set(int(t) for t in task_labels)
        assert len(s) == 1, "Batch multi-task non supportato in questa head."
        return self.forward_single_task(x, next(iter(s)))

    def forward_single_task(self, x: torch.Tensor, task_label: int) -> torch.Tensor:
        return self.heads[str(int(task_label))](x)
