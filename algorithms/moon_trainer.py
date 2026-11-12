"""MOON client-side local training."""

from __future__ import annotations

import copy
from typing import Mapping, Optional
import torch
import torch.nn.functional as F

from utils.moon_helper import MoonFeatureForward
from .base_trainer import BaseTrainer


class MoonTrainer(BaseTrainer):
    """Model-Contrastive Federated Learning trainer."""

    def __init__(
        self,
        mu: float = 1.0,
        temperature: float = 0.5,
        momentum: float = 0.9,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if temperature <= 0:
            raise ValueError("MOON temperature must be greater than zero.")
        self.mu = mu
        self.temperature = temperature
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.last_ce_loss = 0.0
        self.last_con_loss = 0.0


    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)

    def train(
        self,
        model: torch.nn.Module,
        trainloader,
        epochs: int,
        lr: float,
        device: torch.device,
        previous_state_dict: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> float:
        """Train with classification loss plus the MOON contrastive loss."""
        global_model = copy.deepcopy(model)
        has_previous = previous_state_dict is not None
        previous_model = None

        if has_previous:
            previous_model = copy.deepcopy(model)
            previous_model.load_state_dict(previous_state_dict, strict=True)
            previous_model.to(device)
            previous_model.eval()
            for parameter in previous_model.parameters():
                parameter.requires_grad_(False)

        model.to(device)
        global_model.to(device)

        model.train()
        global_model.eval()

        for parameter in global_model.parameters():
            parameter.requires_grad_(False)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss().to(device)

        local_forward = MoonFeatureForward(model)
        global_forward = MoonFeatureForward(global_model)
        previous_forward = MoonFeatureForward(previous_model) if has_previous else None

        total_loss = 0.0
        total_ce_loss = 0.0     
        total_con_loss = 0.0
        num_batches = 0

        try:
            for _ in range(epochs):
                for batch in trainloader:
                    images = batch["img"].to(device)
                    labels = batch["label"].long().to(device)

                    optimizer.zero_grad()

                    _, local_projection, logits = local_forward.forward(images)

                    with torch.no_grad():
                        _, global_projection, _ = global_forward.forward(images)

                    loss_ce = criterion(logits, labels)

                    if has_previous and previous_forward is not None:
                        with torch.no_grad():
                            _, previous_projection, _ = previous_forward.forward(images)

                        pos_sim = F.cosine_similarity(local_projection, global_projection, dim=-1)
                        neg_sim = F.cosine_similarity(local_projection, previous_projection, dim=-1)

                        contrastive_logits = torch.stack((pos_sim, neg_sim), dim=1) / self.temperature
                        contrastive_targets = torch.zeros(images.size(0), dtype=torch.long, device=device)

                        loss_con = criterion(contrastive_logits, contrastive_targets)
                        loss = loss_ce + self.mu * loss_con
                    else:
                        loss_con = torch.tensor(0.0, device=device)
                        loss = loss_ce

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_ce_loss += loss_ce.item()
                    total_con_loss += loss_con.item()
                    num_batches += 1

            # Store the averages at the end of training
            if num_batches > 0:
                self.last_ce_loss = total_ce_loss / num_batches
                self.last_con_loss = total_con_loss / num_batches
        finally:
            local_forward.close()
            global_forward.close()
            if previous_forward is not None:
                previous_forward.close()

        return total_loss / max(num_batches, 1)

    def get_metrics(self) -> dict:
        return {
            "moon_ce_loss": self.last_ce_loss,
            "moon_contrastive_loss": self.last_con_loss,
        }