from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.base_trainer import BaseTrainer
from utils.prototype_helper import FeatureExtractor


class FedPACTrainer(BaseTrainer):
    """FedPAC: Personalized Federated Learning with Feature Alignment and Classifier Collaboration.

    Reference: Xu et al., ICLR 2023.
    Local objective balances cross-entropy classification loss with feature alignment loss
    pulling sample representations toward global class prototypes:
        L = L_CE + lambda_align * MSE(f(x), C_y).
    """

    def __init__(
        self,
        global_prototypes: Optional[Dict[int, torch.Tensor]] = None,
        fedpac_lambda: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.global_prototypes = global_prototypes or {}
        self.fedpac_lambda = float(fedpac_lambda)
        self.extractor = None

    def on_train_start(self, model: nn.Module):
        self.extractor = FeatureExtractor(model)

    def compute_loss(self, model: nn.Module, outputs: torch.Tensor, labels: torch.Tensor, criterion) -> torch.Tensor:
        loss_ce = criterion(outputs, labels)
        if not self.global_prototypes or self.extractor is None or self.fedpac_lambda <= 0.0:
            return loss_ce

        features = self.extractor.features
        if features is None:
            return loss_ce

        # Feature Alignment loss against global class prototypes
        loss_align = 0.0
        aligned_count = 0
        for i in range(len(labels)):
            c = int(labels[i].item())
            if c in self.global_prototypes:
                target_proto = self.global_prototypes[c].to(features.device)
                loss_align += F.mse_loss(features[i], target_proto)
                aligned_count += 1

        if aligned_count > 0:
            loss_align = loss_align / aligned_count
            return loss_ce + self.fedpac_lambda * loss_align

        return loss_ce
