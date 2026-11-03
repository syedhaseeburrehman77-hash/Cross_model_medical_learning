from .base_trainer import BaseTrainer
from utils.prototype_helper import FeatureExtractor
import torch
import torch.nn.functional as F


class FedProtoTrainer(BaseTrainer):
    """FedProto: Federated Prototype Learning across Heterogeneous Clients.

    Tan et al., AAAI 2022.
    Minimizes classification cross-entropy loss augmented with MSE distance
    between local feature embeddings and global class prototypes.
    """

    def __init__(self, global_prototypes=None, proto_lambda=1.0, **kwargs):
        super().__init__(**kwargs)
        self.global_prototypes = global_prototypes or {}
        self.proto_lambda = float(proto_lambda)
        self.extractor = None

    def on_train_start(self, model):
        self.extractor = FeatureExtractor(model)

    def compute_loss(self, model, outputs, labels, criterion):
        loss_ce = criterion(outputs, labels)
        if not self.global_prototypes or self.extractor is None or self.proto_lambda <= 0.0:
            return loss_ce

        features = self.extractor.features
        if features is None:
            return loss_ce

        loss_proto = 0.0
        count = 0
        for i in range(len(labels)):
            c = int(labels[i].item())
            if c in self.global_prototypes:
                target_proto = self.global_prototypes[c].to(features.device)
                loss_proto += F.mse_loss(features[i], target_proto)
                count += 1

        if count > 0:
            loss_proto = loss_proto / count
            return loss_ce + self.proto_lambda * loss_proto

        return loss_ce
