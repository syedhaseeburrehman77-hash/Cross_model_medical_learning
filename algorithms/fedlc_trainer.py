import torch
import torch.nn as nn
from algorithms.base_trainer import BaseTrainer
from utils.fedlc_helper import is_fedlc_local_param, get_head_name


class FedLCTrainer(BaseTrainer):
    """FedLC Trainer: Personalized Federated Learning with Local Calibration.

    Reference: Wang et al., ECCV 2022 (Personalizing Federated Medical Image Segmentation via Local Calibration).
    Adapted here for classification tasks (brain tumor MRI/CT).

    Employs a two-stage local training workflow:
      Stage 1: Freeze shared backbone, train only the client-personalized classification head.
      Stage 2: Freeze head, train the shared backbone to refine global representations.
    """

    def __init__(
        self,
        head_epochs: int = 1,
        alpha: float = 0.0,
        pcs: bool = True,
        hc: bool = True,
        head_module_name: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.head_epochs = int(head_epochs)
        self.alpha = float(alpha)
        self.pcs = bool(pcs)
        self.hc = bool(hc)
        self.head_module_name = head_module_name
        self.last_metrics = {}

    def get_metrics(self) -> dict:
        return self.last_metrics

    def compute_loss(self, model, outputs, labels, criterion) -> torch.Tensor:
        """Compute classification loss. Satisfies BaseTrainer abstract contract."""
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        return criterion(outputs, labels)

    def train(
        self,
        model: nn.Module,
        trainloader,
        local_epochs: int,
        lr: float,
        device: torch.device,
        valloader=None,
        patience: int = 5,
    ) -> float:
        model.to(device)

        # Identify local (head) parameters vs shared (backbone) parameters
        local_params = []
        shared_params = []

        for name, param in model.named_parameters():
            if is_fedlc_local_param(name):
                local_params.append(param)
            else:
                shared_params.append(param)

        # Fallback if no specific local parameters were matched (e.g. unknown arch)
        if not local_params:
            head_name = self.head_module_name or get_head_name(model)
            modules = dict(model.named_modules())
            if head_name in modules:
                local_params = list(modules[head_name].parameters())
                local_ids = {id(p) for p in local_params}
                shared_params = [p for p in model.parameters() if id(p) not in local_ids]
            else:
                local_params = list(model.parameters())
                shared_params = list(model.parameters())

        criterion = nn.CrossEntropyLoss()
        optimizer_head = torch.optim.Adam(local_params, lr=lr, weight_decay=1e-4) if local_params else None
        optimizer_shared = torch.optim.Adam(
            shared_params if shared_params else model.parameters(), lr=lr, weight_decay=1e-4
        )

        def run_step(params, optimizer, epochs):
            total_loss = 0.0
            num_batches = 0

            # Freeze all, then unfreeze only the target params
            for p in model.parameters():
                p.requires_grad = False
            for p in params:
                p.requires_grad = True

            for _ in range(epochs):
                model.train()
                for batch in trainloader:
                    images = batch["img"].to(device)
                    labels = batch["label"].to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = self.compute_loss(model, outputs, labels, criterion)
                    loss.backward()

                    # Guard against NaN/Inf gradients
                    for p in params:
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            raise RuntimeError("NaN/Inf gradients detected in FedLC training")

                    optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1

            return total_loss / max(num_batches, 1)

        # Stage 1: Personal head optimization (freeze backbone, train head)
        stage1_loss = 0.0
        if local_params and self.head_epochs > 0:
            stage1_loss = run_step(local_params, optimizer_head, self.head_epochs)

        # Stage 2: Shared backbone optimization (freeze head, train backbone)
        stage2_loss = run_step(
            shared_params if shared_params else list(model.parameters()),
            optimizer_shared,
            local_epochs,
        )

        # Re-enable requires_grad for all parameters
        for p in model.parameters():
            p.requires_grad = True

        self.last_metrics = {
            "fedlc_head_loss": stage1_loss,
            "fedlc_train_loss": stage2_loss,
        }
        return stage2_loss
