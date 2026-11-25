"""FedCM-MOON client-side local trainer.

Extends MOON (Model-Contrastive Federated Learning) with Modality Group-Contrastive Alignment:
L = CE(h(z), y) + mu * L_con(z; pos = z_glob, neg = z_prev) + lambda_t * L_con(z; pos = z_grp, neg = z_prev)
"""

from __future__ import annotations

import copy
from typing import Mapping, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.moon_helper import MoonFeatureForward
from .base_trainer import BaseTrainer


class FedCMMoonTrainer(BaseTrainer):
    """Federated Cross-Modal MOON Trainer with Global and Group Contrastive Alignment."""

    def __init__(
        self,
        group_model_state: Optional[Mapping[str, torch.Tensor]] = None,
        mu: float = 1.0,
        lambda_t: float = 0.0,
        temperature: float = 0.5,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if temperature <= 0:
            raise ValueError("[FedCM-MOON Error] Temperature must be greater than zero.")
        self.group_model_state = group_model_state
        self.mu = float(mu)
        self.lambda_t = float(lambda_t)
        self.temperature = float(temperature)
        self.weight_decay = float(weight_decay)

        self.last_ce_loss = 0.0
        self.last_glob_con_loss = 0.0
        self.last_grp_con_loss = 0.0

    def compute_loss(self, model: nn.Module, outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """Satisfies BaseTrainer contract."""
        return criterion(outputs, labels)

    def train(
        self,
        model: torch.nn.Module,
        trainloader,
        epochs: int,
        lr: float,
        device: torch.device,
        previous_state_dict: Optional[Mapping[str, torch.Tensor]] = None,
        global_state_dict: Optional[Mapping[str, torch.Tensor]] = None,
        group_state_dict: Optional[Mapping[str, torch.Tensor]] = None,
        **kwargs,
    ) -> float:
        """Local training loop with global and group model contrastive alignment."""
        # 1. Prepare frozen Global Model (eval mode, no grad)
        global_model = copy.deepcopy(model)
        if global_state_dict is not None:
            global_model.load_state_dict(global_state_dict, strict=True)
        global_model.to(device)
        global_model.eval()
        for p in global_model.parameters():
            p.requires_grad_(False)

        # 2. Prepare frozen Group Model (eval mode, no grad)
        grp_state = group_state_dict or self.group_model_state
        group_model = copy.deepcopy(model)
        if grp_state is not None:
            group_model.load_state_dict(grp_state, strict=True)
        group_model.to(device)
        group_model.eval()
        for p in group_model.parameters():
            p.requires_grad_(False)

        # 3. Prepare frozen Previous Local Model (eval mode, no grad)
        has_previous = previous_state_dict is not None
        previous_model = None
        if has_previous:
            previous_model = copy.deepcopy(model)
            previous_model.load_state_dict(previous_state_dict, strict=True)
            previous_model.to(device)
            previous_model.eval()
            for p in previous_model.parameters():
                p.requires_grad_(False)

        # 4. Trainable local model
        model.to(device)
        model.train()

        # 5. Fresh Adam optimizer created at the start of the round (as specified in methodology)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=self.weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss().to(device)

        # 6. Feature extractor forward hooks (reusing MoonFeatureForward)
        local_forward = MoonFeatureForward(model)
        global_forward = MoonFeatureForward(global_model)
        group_forward = MoonFeatureForward(group_model)
        previous_forward = MoonFeatureForward(previous_model) if has_previous else None

        total_loss = 0.0
        total_ce_loss = 0.0
        total_glob_loss = 0.0
        total_grp_loss = 0.0
        num_batches = 0

        try:
            for _ in range(epochs):
                for batch in trainloader:
                    images = batch["img"].to(device)
                    labels = batch["label"].long().to(device)

                    optimizer.zero_grad()

                    # Trainable model forward pass
                    _, local_proj, logits = local_forward.forward(images)

                    # Frozen anchor forward passes under torch.no_grad()
                    with torch.no_grad():
                        _, global_proj, _ = global_forward.forward(images)
                        _, group_proj, _ = group_forward.forward(images)

                    loss_ce = criterion(logits, labels)

                    if has_previous and previous_forward is not None:
                        with torch.no_grad():
                            _, previous_proj, _ = previous_forward.forward(images)

                        # Negative similarity: against previous local model
                        neg_sim = F.cosine_similarity(local_proj, previous_proj, dim=-1)
                        targets = torch.zeros(images.size(0), dtype=torch.long, device=device)

                        # 1. Global Contrastive Loss: pull toward W_g, push away from W_prev
                        pos_sim_glob = F.cosine_similarity(local_proj, global_proj, dim=-1)
                        logits_glob = torch.stack((pos_sim_glob, neg_sim), dim=1) / self.temperature
                        loss_con_glob = criterion(logits_glob, targets)

                        # 2. Group Contrastive Loss: pull toward W_grp, push away from W_prev
                        if self.lambda_t > 0.0:
                            pos_sim_grp = F.cosine_similarity(local_proj, group_proj, dim=-1)
                            logits_grp = torch.stack((pos_sim_grp, neg_sim), dim=1) / self.temperature
                            loss_con_grp = criterion(logits_grp, targets)
                        else:
                            loss_con_grp = torch.tensor(0.0, device=device)

                        loss = loss_ce + (self.mu * loss_con_glob) + (self.lambda_t * loss_con_grp)
                    else:
                        loss_con_glob = torch.tensor(0.0, device=device)
                        loss_con_grp = torch.tensor(0.0, device=device)
                        loss = loss_ce

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_ce_loss += loss_ce.item()
                    total_glob_loss += loss_con_glob.item()
                    total_grp_loss += loss_con_grp.item()
                    num_batches += 1

            if num_batches > 0:
                self.last_ce_loss = total_ce_loss / num_batches
                self.last_glob_con_loss = total_glob_loss / num_batches
                self.last_grp_con_loss = total_grp_loss / num_batches

        finally:
            local_forward.close()
            global_forward.close()
            group_forward.close()
            if previous_forward is not None:
                previous_forward.close()

        return total_loss / max(num_batches, 1)

    def get_metrics(self) -> dict:
        return {
            "moon_ce_loss": self.last_ce_loss,
            "moon_glob_con_loss": self.last_glob_con_loss,
            "moon_grp_con_loss": self.last_grp_con_loss,
            "moon_lambda_t": self.lambda_t,
        }
