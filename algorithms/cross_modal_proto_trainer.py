from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from utils.prototype_helper import FeatureExtractor


class CrossModalProtoTrainer(BaseTrainer):
    """Client-side Trainer for Cross-Modal Prototype Alignment with Local Batch Normalization."""

    def __init__(
        self,
        opposite_master_prototypes: Optional[Dict[int, torch.Tensor]] = None,
        transfer_weights: Optional[Dict[int, float]] = None,
        align_lambda: float = 0.5,
        is_warmup: bool = False,
        client_id: int = 0,
        modality: str = "MRI",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.opposite_master_prototypes = opposite_master_prototypes or {}
        self.transfer_weights = transfer_weights or {}
        self.align_lambda = float(align_lambda)
        self.is_warmup = bool(is_warmup)
        self.client_id = client_id
        self.modality = modality
        self.extractor: Optional[FeatureExtractor] = None

        # Metrics tracking
        self.last_ce_loss = 0.0
        self.last_align_loss = 0.0

    def on_train_start(self, model: nn.Module):
        """Hook into the model to capture latent feature representations before classification."""
        self.extractor = FeatureExtractor(model)

        if self.modality == "CT":
            opp_mod = "MRI"
        elif self.modality == "MRI":
            opp_mod = "CT"
        elif self.modality == "Clinical":
            opp_mod = "Dermoscopy"
        elif self.modality == "Dermoscopy":
            opp_mod = "Clinical"
        else:
            opp_mod = "Opposite"
        if self.is_warmup:
            print(
                f"[CrossModal-Client {self.client_id} ({self.modality})] "
                f"==> WARM-UP PHASE: Standard classification training (Alignment disabled)."
            )
        else:
            active_weights = {
                c: round(self.transfer_weights.get(c, 0.0), 4)
                for c in sorted(list(self.opposite_master_prototypes.keys()))
            }
            print(
                f"[CrossModal-Client {self.client_id} ({self.modality})] "
                f"==> ALIGNMENT ACTIVE: Opposite ({opp_mod}) Prototypes: {len(self.opposite_master_prototypes)}, "
                f"Transfer Weights: {active_weights}, Lambda={self.align_lambda}"
            )

    def compute_loss(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """Compute the combined classification loss + directional prototype alignment loss."""
        loss_ce = criterion(outputs, labels)
        self.last_ce_loss = loss_ce.item()

        # If in warm-up or no opposite prototypes available or alignment disabled
        if (
            self.is_warmup
            or not self.opposite_master_prototypes
            or self.extractor is None
            or self.align_lambda <= 0.0
        ):
            self.last_align_loss = 0.0
            return loss_ce

        features = self.extractor.features
        if features is None:
            self.last_align_loss = 0.0
            return loss_ce

        # Compute mini-batch class prototypes and align with opposite-modality master prototypes.
        # Formula: L_align = Σ_c [ g_{k,c} * (1 - cos(p_local_c, P_opp_c)) ] / (Σ_c g_{k,c} + ε)
        # Only classes present in both the mini-batch and the opposite-modality prototypes are included.

        eps = 1e-8
        loss_align = torch.tensor(0.0, device=features.device)
        weight_sum = 0.0

        # Collect all unique class labels in this mini-batch
        batch_classes = labels.unique().tolist()

        for c_val in batch_classes:
            c = int(c_val)
            if c not in self.opposite_master_prototypes:
                continue

            # Compute differentiable mini-batch class prototype (mean of same-class features)
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            local_proto = features[mask].mean(dim=0)  # (feature_dim,)

            target_proto = self.opposite_master_prototypes[c].to(features.device).detach()
            weight = float(self.transfer_weights.get(c, 0.0))

            # Cosine distance = 1 - cosine_similarity
            cos_sim = F.cosine_similarity(
                local_proto.unsqueeze(0), target_proto.unsqueeze(0), dim=1
            )
            cos_dist = 1.0 - cos_sim.squeeze(0)

            loss_align = loss_align + (weight * cos_dist)
            weight_sum += weight

        if weight_sum > 0.0:
            loss_align = loss_align / (weight_sum + eps)
            self.last_align_loss = loss_align.item()
            total_loss = loss_ce + (self.align_lambda * loss_align)
            return total_loss

        self.last_align_loss = 0.0
        return loss_ce

    def get_metrics(self) -> Dict[str, float]:
        """Return trainer-specific monitoring metrics."""
        return {
            "ce_loss": self.last_ce_loss,
            "align_loss": self.last_align_loss,
            "is_warmup": 1.0 if self.is_warmup else 0.0,
        }
