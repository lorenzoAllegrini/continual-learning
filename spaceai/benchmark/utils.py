import torch
from torch.utils.data import DataLoader
from contextlib import nullcontext
from avalanche.models.dynamic_modules import avalanche_model_adaptation
from spaceai.data.utils import seq_collate_fn
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from tqdm import tqdm
import copy

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
        patience_before_stopping=None,
        min_delta=None,
        restore_best=False,
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
        
        self.patience_before_stopping = patience_before_stopping
        self.min_delta = min_delta
        self.restore_best = restore_best

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
    
    def train_experience(
        self,
        data,
        task,
        *,
        eval_data=None,
    ):

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
        best_val_loss = float("inf")
        min_delta_ = self.min_delta if self.min_delta is not None else 0.0
        epochs_since_improvement = 0
        if self.restore_best:
            best_model = copy.deepcopy(self.model.state_dict())

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
                if eval_data is not None:
                    val_loss = self.evaluate_experience(eval_data, task)
                else:
                    val_loss = avg_loss

                if val_loss < best_val_loss - min_delta_:
                    best_val_loss = val_loss
                    epochs_since_improvement = 0
                    if self.restore_best:
                        best_model = copy.deepcopy(self.model.state_dict())
                else:
                    epochs_since_improvement += 1

                pbar.set_description(
                    f"Epoch [{epoch + 1}/{self.train_epochs}] Loss: {avg_loss:.4f} Val: {val_loss:.4f}"
                )
                pbar.update(1)
                if (
                    self.patience_before_stopping is not None
                    and epochs_since_improvement >= self.patience_before_stopping
                ):
                    break

        if self.restore_best:
            self.model.load_state_dict(best_model)
            pbar.set_description(
                f"Epoch [{epoch + 1}/{self.train_epochs}] Loss: {avg_loss:.4f}"
            )
            pbar.update(1)
                
    # comodo helper per uno stream di experience (opzionale)
    def train_stream(self, train_stream):
        for exp in train_stream:
            self.train_experience(exp)

__all__ = ["CLTrainer"]

"""diff --git a/spaceai/benchmark/utils.py b/spaceai/benchmark/utils.py
index 2d955905ac12ef2e3d07369200ab8acd2f169ea9..c8669a75c66a9be5496103e7adfdf0c596fb64f1 100644
--- a/spaceai/benchmark/utils.py
+++ b/spaceai/benchmark/utils.py
@@ -1,32 +1,30 @@
+import copy
 import torch
 from torch.utils.data import DataLoader
 from contextlib import nullcontext
-from avalanche.models.dynamic_modules import avalanche_model_adaptation
 from spaceai.data.utils import seq_collate_fn
-from avalanche.benchmarks.utils import AvalancheDataset
-from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
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
diff --git a/spaceai/benchmark/utils.py b/spaceai/benchmark/utils.py
index 2d955905ac12ef2e3d07369200ab8acd2f169ea9..c8669a75c66a9be5496103e7adfdf0c596fb64f1 100644
--- a/spaceai/benchmark/utils.py
+++ b/spaceai/benchmark/utils.py
@@ -82,78 +80,129 @@ class CLTrainer:
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
     
-    def train_experience(self, data, task):
+    def train_experience(
+        self,
+        data,
+        task,
+        *,
+        eval_data=None,
+        patience_before_stopping=None,
+        min_delta=None,
+        restore_best=True,
+    ):
+
+        try:
+            from avalanche.benchmarks.utils import AvalancheDataset
+            from avalanche.benchmarks.scenarios.dataset_scenario import (
+                benchmark_from_datasets,
+            )
+            from avalanche.models.dynamic_modules import avalanche_model_adaptation
+        except ImportError as exc:  # pragma: no cover - safeguard if avalanche missing
+            raise ImportError(
+                "avalanche is required for train_experience but is not installed"
+            ) from exc
 
         # 1) Adattamento (come Naive)
-        train_ad = AvalancheDataset(data, collate_fn=seq_collate_fn(n_inputs=2, mode="time"))
-        train_ad = train_ad.update_data_attribute("targets_task_labels", [task] * len(data))
+        train_ad = AvalancheDataset(
+            data, collate_fn=seq_collate_fn(n_inputs=2, mode="time")
+        )
+        train_ad = train_ad.update_data_attribute(
+            "targets_task_labels", [task] * len(data)
+        )
         benchmark = benchmark_from_datasets(train=[train_ad], test=[])
         avalanche_model_adaptation(self.model, benchmark.train_stream[0])
-      
+
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
 
+        best_val_loss = float("inf")
+        min_delta_ = min_delta if min_delta is not None else 0.0
+        epochs_since_improvement = 0
+        if restore_best:
+            best_model = copy.deepcopy(self.model.state_dict())
+
         with tqdm(total=self.train_epochs) as pbar:
-            
+
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
+                if eval_data is not None:
+                    val_loss = self.evaluate_experience(eval_data, task)
+                else:
+                    val_loss = avg_loss
+
+                if val_loss < best_val_loss - min_delta_:
+                    best_val_loss = val_loss
+                    epochs_since_improvement = 0
+                    if restore_best:
+                        best_model = copy.deepcopy(self.model.state_dict())
+                else:
+                    epochs_since_improvement += 1
+
                 pbar.set_description(
-                    f"Epoch [{epoch + 1}/{self.train_epochs}] Loss: {avg_loss:.4f}"
+                    f"Epoch [{epoch + 1}/{self.train_epochs}] Loss: {avg_loss:.4f} Val: {val_loss:.4f}"
                 )
                 pbar.update(1)
+                if (
+                    patience_before_stopping is not None
+                    and epochs_since_improvement >= patience_before_stopping
+                ):
+                    break
+
+        if restore_best:
+            self.model.load_state_dict(best_model)
                 
     # comodo helper per uno stream di experience (opzionale)
     def train_stream(self, train_stream):
         for exp in train_stream:
             self.train_experience(exp)
 
 __all__ = ["CLTrainer"]
"""