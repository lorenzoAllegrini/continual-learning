import torch
from torch.utils.data import DataLoader
from contextlib import nullcontext
from avalanche.models.dynamic_modules import avalanche_model_adaptation
from spaceai.data.utils import seq_collate_fn
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from tqdm import tqdm

def snapshot_params(model):
    # copia leggera dei tensori dei pesi per calcolare le differenze dopo lo step
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

def grad_report(model, prefix=""):
    lines = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            lines.append(f"[FROZEN] {prefix}{n}")
            continue
        g = p.grad
        if g is None:
            lines.append(f"[NO-GRAD] {prefix}{n}")
        else:
            lines.append(f"[GRAD]    {prefix}{n}: ||g||={g.data.norm().item():.4e}")
    return "\n".join(lines)

def delta_report(model, before, prefix=""):
    # mostra quanto è cambiato ogni parametro dopo optimizer.step()
    lines = []
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n not in before:
                lines.append(f"[NEW]  {prefix}{n} (aggiunto dopo lo snapshot)")
                continue
            if not p.requires_grad:
                lines.append(f"[FROZEN]{prefix}{n}")
                continue
            delta = (p - before[n]).norm().item()
            lines.append(f"[Δ]     {prefix}{n}: ||Δ||={delta:.4e}")
    return "\n".join(lines)



class CLTrainer:
    def __init__(
        self,
        model,
        optimizer_factory,
        criterion,
        *,
        device="cpu",
        train_epochs=20,
        train_mb_size=1024,
        eval_mb_size=64,
        collate_fn=None,
        num_workers=0,
        grad_clip=None,
        use_amp=False,
        washout=249,
    ):
        self.model = model
        self.optimizer_factory = optimizer_factory
        self.criterion = criterion
        self.device = torch.device(device)

        # iperparametri "alla Naive"
        self.train_epochs = train_epochs
        self.train_mb_size = train_mb_size
        self.eval_mb_size = eval_mb_size

        # dataloader / training extras
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.grad_clip = grad_clip
        # abilita AMP solo se su cuda
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.washout = washout

    @torch.no_grad()
    def evaluate_experience(self, data, task):
        self.model.eval()
        dl = DataLoader(
            data,
            batch_size=self.eval_mb_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=seq_collate_fn(n_inputs=2, mode="batch"),
        )
        total, n = 0.0, 0
        for inputs, targets in dl:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs, task)
            outputs, targets = self._apply_washout(outputs, targets)
            loss = self.criterion(outputs, targets)
            bs = inputs.size(0)
            total += loss.item() * bs
            n += bs
        return total / max(n, 1)

    def _apply_washout(self, inputs: torch.Tensor, targets: torch.Tensor):
        if self.washout is not None:
            targets = targets[self.washout :]
            inputs = inputs[-len(targets) :]
        return inputs, targets
    
    def train_experience(self, data, task):

        # 1) Adattamento (come Naive)
        train_ad = AvalancheDataset(data, collate_fn=seq_collate_fn(n_inputs=2, mode="time"))
        train_ad = train_ad.update_data_attribute("targets_task_labels", [task] * len(data))
        benchmark = benchmark_from_datasets(train=[train_ad], test=[])
        avalanche_model_adaptation(self.model, benchmark.train_stream[0])
      
        # 2) Dataloader
        dl = DataLoader(
            data,
            batch_size=64,
            shuffle=True,
            collate_fn=seq_collate_fn(n_inputs=2, mode="batch"),
        )

        optimizer = self.optimizer_factory(self.model)
        # 3) Loop
        self.model.train()

        with tqdm(total=self.train_epochs) as pbar:
            
            for epoch in range(self.train_epochs):
                epoch_loss = 0.0
                n_samples = 0
                for inputs, targets in dl:
                    before = snapshot_params(self.model)
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs, task)
                    #print(outputs.shape)
                    outputs, targets = self._apply_washout(outputs, targets)
                    loss = self.criterion(outputs, targets)

                    loss.backward()
                    optimizer.step()
                    #print(grad_report(self.model))
                    #print(delta_report(self.model, before))
                    epoch_loss += loss.item() * inputs.size(0)
                    n_samples += inputs.size(0)

                avg_loss = epoch_loss / max(n_samples, 1)
                pbar.set_description(
                    f"Epoch [{epoch + 1}/{self.train_epochs}] Loss: {avg_loss:.4f}"
                )
                pbar.update(1)
                
    # comodo helper per uno stream di experience (opzionale)
    def train_stream(self, train_stream):
        for exp in train_stream:
            self.train_experience(exp)

__all__ = ["CLTrainer"]
