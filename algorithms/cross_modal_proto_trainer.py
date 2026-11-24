from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from utils.prototype_helper import FeatureExtractor

_OPPOSITE_MODALITY_MAP = {
    "CT": "MRI",
    "MRI": "CT",
    "Clinical": "Dermoscopy",
    "Dermoscopy": "Clinical",
}


class CrossModalProtoTrainer(BaseTrainer):
    """Client-side Trainer for Cross-Modal Prototype Alignment."""

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
        self.client_id = int(client_id)
        self.modality = modality
        self.extractor: Optional[FeatureExtractor] = None
        self._device_target_prototypes: Dict[int, torch.Tensor] = {}

        self.last_ce_loss = 0.0
        self.last_align_loss = 0.0

    def on_train_start(self, model: nn.Module):
        """Attach feature hook and stage normalized opposite prototypes on the model's device."""
        if self.modality not in _OPPOSITE_MODALITY_MAP:
            raise ValueError(
                f"[CrossModalProtoTrainer Error] Unknown modality '{self.modality}' for client {self.client_id}."
            )
        opp_mod = _OPPOSITE_MODALITY_MAP[self.modality]
        self.extractor = FeatureExtractor(model)

        if self.is_warmup:
            self._device_target_prototypes = {}
            print(
                f"[CrossModal-Client {self.client_id} ({self.modality})] "
                f"==> WARM-UP PHASE: Standard classification training (Alignment disabled)."
            )
            return

        if not self.opposite_master_prototypes:
            raise RuntimeError(
                f"[CrossModalProtoTrainer Error] Client {self.client_id} ({self.modality}) is in alignment phase "
                f"but received empty opposite_master_prototypes."
            )
        if not self.transfer_weights:
            raise RuntimeError(
                f"[CrossModalProtoTrainer Error] Client {self.client_id} ({self.modality}) is in alignment phase "
                f"but received empty transfer_weights."
            )
        if self.align_lambda <= 0.0:
            raise ValueError(
                f"[CrossModalProtoTrainer Error] align_lambda must be > 0 during alignment phase, got {self.align_lambda}."
            )

        model_device = next(model.parameters()).device
        self._device_target_prototypes = {
            int(c): F.normalize(p.detach().float().to(model_device), p=2, dim=0, eps=1e-8)
            for c, p in self.opposite_master_prototypes.items()
        }

        active_weights = {
            c: round(float(self.transfer_weights[c]), 4)
            for c in sorted(self._device_target_prototypes.keys())
        }
        print(
            f"[CrossModal-Client {self.client_id} ({self.modality})] "
            f"==> ALIGNMENT ACTIVE: Opposite ({opp_mod}) Prototypes: {len(self._device_target_prototypes)}, "
            f"Transfer Weights: {active_weights}, Lambda={self.align_lambda}"
        )

    def remove_hooks(self):
        """Detach feature extractor hook cleanly after training."""
        if self.extractor is not None:
            self.extractor.remove()
            self.extractor = None

    def compute_loss(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """Compute classification cross-entropy + directional unit-sphere prototype alignment loss."""
        loss_ce = criterion(outputs, labels)
        self.last_ce_loss = loss_ce.item()

        if self.is_warmup:
            self.last_align_loss = 0.0
            return loss_ce

        if self.extractor is None or self.extractor.features is None:
            raise RuntimeError(
                f"[CrossModalProtoTrainer Error] FeatureExtractor failed to capture features on client {self.client_id}."
            )
        if not self._device_target_prototypes:
            raise RuntimeError(
                f"[CrossModalProtoTrainer Error] Target prototypes were not initialized on client {self.client_id}."
            )

        features = self.extractor.features
        loss_align = torch.tensor(0.0, device=features.device)
        aligned_count = 0

        for c_val in labels.unique().tolist():
            c = int(c_val)
            if c not in self._device_target_prototypes:
                continue
            if c not in self.transfer_weights:
                raise KeyError(
                    f"[CrossModalProtoTrainer Error] Missing transfer weight for class {c} on client {self.client_id}."
                )

            mask = (labels == c)
            class_feats = F.normalize(features[mask], p=2, dim=1, eps=1e-8)
            local_proto = F.normalize(class_feats.mean(dim=0), p=2, dim=0, eps=1e-8)

            target_proto = self._device_target_prototypes[c]
            weight = float(self.transfer_weights[c])

            cos_sim = F.cosine_similarity(
                local_proto.unsqueeze(0), target_proto.unsqueeze(0), dim=1
            )
            cos_dist = 1.0 - cos_sim.squeeze(0)

            loss_align = loss_align + (weight * cos_dist)
            aligned_count += 1

        if aligned_count > 0:
            loss_align = loss_align / float(aligned_count)
            self.last_align_loss = loss_align.item()
            return loss_ce + (self.align_lambda * loss_align)

        self.last_align_loss = 0.0
        return loss_ce

    def get_metrics(self) -> Dict[str, float]:
        """Return trainer-specific monitoring metrics."""
        return {
            "ce_loss": self.last_ce_loss,
            "align_loss": self.last_align_loss,
            "is_warmup": 1.0 if self.is_warmup else 0.0,
        }
