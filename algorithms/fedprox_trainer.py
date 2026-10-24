import torch
from .base_trainer import BaseTrainer

class FedProxTrainer(BaseTrainer):
    """FedAvg + proximal term penalizing drift from the global model."""

    def __init__(self, proximal_mu: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.proximal_mu = proximal_mu
        self.global_params = None
        self.last_proximal_term = 0.0

    def on_train_start(self, model):
        # Snapshot global params BEFORE local training mutates them
        self.global_params = [p.detach().clone() for p in model.parameters()]

    def compute_loss(self, model, outputs, labels, criterion):
        loss = criterion(outputs, labels)

        proximal_term = sum(
            (local_p - global_p).pow(2).sum()
            for local_p, global_p in zip(model.parameters(), self.global_params)
        )
        self.last_proximal_term = float(proximal_term.item())
        loss += (self.proximal_mu / 2) * proximal_term
        return loss

    def get_metrics(self) -> dict:
        return {"proximal_penalty": self.last_proximal_term}