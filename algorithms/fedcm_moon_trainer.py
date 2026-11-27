"""FedCM-MOON client-side local trainer.

Extends MOON (Model-Contrastive Federated Learning) with Class-Wise Modality Prototype Alignment:
L = CE(logits, y) + mu * L_con(z; pos = z_glob, neg = z_prev) + lambda_t * L_align(p_kc, P_c^{opp})
"""

from __future__ import annotations

import copy
from typing import Dict, Mapping, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.moon_helper import MoonFeatureForward
from utils.prototype_helper import FeatureExtractor
from .base_trainer import BaseTrainer


class FedCMMoonTrainer(BaseTrainer):
    """Federated Cross-Modal MOON Trainer with Dual Hybrid Objective:
    MOON Contrastive Alignment + Class-Wise Directional Prototype Alignment.
    """

    def __init__(
        self,
        opposite_master_prototypes: Optional[Dict[int, torch.Tensor]] = None,
        transfer_weights: Optional[Dict[int, float]] = None,
        group_model_state: Optional[Mapping[str, torch.Tensor]] = None,
        mu: float = 1.0,
        lambda_t: float = 0.0,
        temperature: float = 0.5,
        momentum: float = 0.9,
        weight_decay: float = 1e-5,
        client_id: int = 0,
        modality: str = "MRI",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if temperature <= 0:
            raise ValueError("[FedCM-MOON Error] Temperature must be greater than zero.")
        self.opposite_master_prototypes = opposite_master_prototypes or {}
        self.transfer_weights = transfer_weights or {}
        self.group_model_state = group_model_state
        self.mu = float(mu)
        self.lambda_t = float(lambda_t)
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.client_id = int(client_id)
        self.modality = modality

        self._device_target_prototypes: Dict[int, torch.Tensor] = {}

        self.last_ce_loss = 0.0
        self.last_glob_con_loss = 0.0
        self.last_align_loss = 0.0

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
        """Local training loop with MOON contrastive alignment and class-wise prototype alignment."""
        # 1. Prepare frozen Global Model (eval mode, no grad)
        global_model = copy.deepcopy(model)
        if global_state_dict is not None:
            global_model.load_state_dict(global_state_dict, strict=True)
        global_model.to(device)
        global_model.eval()
        for p in global_model.parameters():
            p.requires_grad_(False)

        # 2. Stage L2-normalized opposite master prototypes on device (Idea C)
        self._device_target_prototypes = {
            int(c): F.normalize(p.detach().float().to(device), p=2, dim=0, eps=1e-8)
            for c, p in self.opposite_master_prototypes.items()
        }

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

        # 5. Stable SGD optimizer with momentum (matching simple MOON)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss().to(device)

        # 6. Feature extractor forward hooks (local, global, previous, and classification prototype extractor)
        local_forward = MoonFeatureForward(model)
        global_forward = MoonFeatureForward(global_model)
        previous_forward = MoonFeatureForward(previous_model) if has_previous else None
        # Hook pre-dropout 256-dim features for fedcm_moon so train() and eval() prototypes match.
        proto_extractor = FeatureExtractor(model, before_dropout=True)

        total_loss = 0.0
        total_ce_loss = 0.0
        total_glob_loss = 0.0
        total_align_loss = 0.0
        num_batches = 0

        print(
            f"[FedCM-MOON Client {self.client_id} ({self.modality})] "
            f"Training {epochs} local epochs ({len(trainloader.dataset)} samples, {len(trainloader)} batches/epoch, lr={lr})..."
        )

        try:
            for epoch in range(1, epochs + 1):
                for batch in trainloader:
                    images = batch["img"].to(device)
                    labels = batch["label"].long().to(device)

                    optimizer.zero_grad()

                    # Trainable model forward pass:
                    # - local_forward captures 1280-dim backbone features and calculates 256-dim MOON projection z
                    # - proto_extractor captures 256-dim pre-classifier feature representations
                    _, local_proj, logits = local_forward.forward(images)
                    proto_features = proto_extractor.features

                    # 1. Classification Cross-Entropy Loss
                    loss_ce = criterion(logits, labels)

                    # 2. MOON Model-Contrastive Loss: pull toward W_g, push away from W_prev
                    if has_previous and previous_forward is not None:
                        with torch.no_grad():
                            _, global_proj, _ = global_forward.forward(images)
                            _, previous_proj, _ = previous_forward.forward(images)

                        pos_sim_glob = F.cosine_similarity(local_proj, global_proj, dim=-1)
                        neg_sim = F.cosine_similarity(local_proj, previous_proj, dim=-1)

                        logits_glob = torch.stack((pos_sim_glob, neg_sim), dim=1) / self.temperature
                        targets = torch.zeros(images.size(0), dtype=torch.long, device=device)
                        loss_con_glob = criterion(logits_glob, targets)
                    else:
                        loss_con_glob = torch.tensor(0.0, device=device)

                    # 3. Idea C: Class-Wise Modality Prototype Alignment Loss
                    if self.lambda_t > 0.0 and self._device_target_prototypes and proto_features is not None:
                        loss_align_num = torch.tensor(0.0, device=device)
                        aligned_class_count = 0
                        for c_val in labels.unique().tolist():
                            c = int(c_val)
                            if c not in self._device_target_prototypes:
                                continue

                            mask = (labels == c)
                            if not mask.any():
                                continue

                            p_kc = proto_features[mask].mean(dim=0)
                            P_opp = self._device_target_prototypes[c]

                            g_kc = float(self.transfer_weights.get(c, 1.0)) if self.transfer_weights else 1.0
                            cos_sim = F.cosine_similarity(p_kc.unsqueeze(0), P_opp.unsqueeze(0), dim=1).squeeze(0)
                            cos_dist = 1.0 - cos_sim

                            # Directional gating: g_kc preserves absolute magnitude in [0, 1]
                            loss_align_num = loss_align_num + (g_kc * cos_dist)
                            aligned_class_count += 1

                        if aligned_class_count > 0:
                            # Average across classes in batch without dividing by sum(g_kc)
                            # so that the absolute gating magnitude of (1 - Conf_client) is preserved
                            loss_align = loss_align_num / float(aligned_class_count)
                        else:
                            loss_align = torch.tensor(0.0, device=device)
                    else:
                        loss_align = torch.tensor(0.0, device=device)

                    # Total Dual Hybrid Objective
                    loss = loss_ce + (self.mu * loss_con_glob) + (self.lambda_t * loss_align)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_ce_loss += loss_ce.item()
                    total_glob_loss += loss_con_glob.item()
                    total_align_loss += loss_align.item()
                    num_batches += 1

            if num_batches > 0:
                self.last_ce_loss = total_ce_loss / num_batches
                self.last_glob_con_loss = total_glob_loss / num_batches
                self.last_align_loss = total_align_loss / num_batches

        finally:
            local_forward.close()
            global_forward.close()
            if previous_forward is not None:
                previous_forward.close()
            proto_extractor.remove()

        return total_loss / max(num_batches, 1)

    def get_metrics(self) -> dict:
        return {
            "moon_ce_loss": self.last_ce_loss,
            "moon_con_loss": self.last_glob_con_loss,
            "moon_align_loss": self.last_align_loss,
            "moon_lambda_t": self.lambda_t,
        }
