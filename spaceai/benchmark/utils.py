import torch
from torch.utils.data import DataLoader
from contextlib import nullcontext
from avalanche.models.dynamic_modules import avalanche_model_adaptation
from spaceai.data.utils import seq_collate_fn
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
class CLTrainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        *,
        device="cpu",
        train_epochs=1,
        train_mb_size=64,
        eval_mb_size=64,
        collate_fn=None,
        num_workers=0,
        grad_clip=None,
        use_amp=False
    ):
        self.model = model
        self.optimizer = optimizer
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

    @torch.no_grad()
    def evaluate_experience(self, experience):
        self.model.eval()
        dl = DataLoader(
            experience.dataset,
            batch_size=self.eval_mb_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        total, n = 0.0, 0
        for x, y, t in dl:
            x = x.to(self.device); y = y.to(self.device)
            out = self.model(x, t)
            loss = self.criterion(out.squeeze(-1), y.squeeze(-1))
            bs = x.size(0)
            total += loss.item() * bs
            n += bs
        return total / max(n, 1)

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

        # 3) Loop
        self.model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        autocast_ctx = torch.cuda.amp.autocast if self.use_amp else nullcontext

        for _ in range(self.train_epochs):
        
            for x, y in dl:
             
                x = x.to(self.device); y = y.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                with autocast_ctx():
                    out = self.model(x, task)
                    loss = self.criterion(out.squeeze(-1), y.squeeze(-1))

                if self.use_amp:
                    scaler.scale(loss).backward()
                    if self.grad_clip is not None:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    scaler.step(self.optimizer); scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

    # comodo helper per uno stream di experience (opzionale)
    def train_stream(self, train_stream):
        for exp in train_stream:
            self.train_experience(exp)

all = ["CLTrainer"]