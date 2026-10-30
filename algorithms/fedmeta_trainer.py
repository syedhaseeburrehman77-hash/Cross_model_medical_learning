"""FedMeta: Federated Meta-Learning Client Trainers.

Supports:
- FedAvg-Meta: Standard local SGD on support set + 1-step meta-testing adaptation.
- FedMeta-MAML: Inner-loop adaptation on support set + outer-loop meta-gradients on query set.
- FedMeta-Meta-SGD: Inner-loop adaptation with learnable vector step size alpha + meta-gradients.
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .base_trainer import BaseTrainer
from utils.metrics import evaluate_model


def split_support_query(dataloader: DataLoader, support_ratio: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Split a DataLoader's dataset into support and query DataLoaders.
    
    Uses drop_last=True to prevent 1-sample batches that crash BatchNorm in training mode.
    """
    dataset = dataloader.dataset
    total_len = len(dataset)
    sup_len = max(1, int(total_len * support_ratio))
    qry_len = total_len - sup_len

    indices = list(range(total_len))
    batch_size = dataloader.batch_size or 32

    # Drop remainder batches only if we have more than 1 batch to prevent single-sample BatchNorm crash
    drop_sup = sup_len > batch_size
    drop_qry = qry_len > batch_size

    sup_loader = DataLoader(
        Subset(dataset, indices[:sup_len]),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_sup,
    )
    qry_loader = DataLoader(
        Subset(dataset, indices[sup_len:]),
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_qry,
    )
    return sup_loader, qry_loader


class FedMetaBaseTrainer(BaseTrainer):
    """Base class for FedMeta trainers with support/query splitting and meta-testing."""

    def __init__(
        self,
        alpha: float = 0.01,
        gradient_step: int = 5,
        support_ratio: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.gradient_step = int(gradient_step)
        self.support_ratio = float(support_ratio)
        self.last_gradients: Optional[Dict[str, torch.Tensor]] = None

    def compute_loss(self, model, outputs, labels, criterion) -> torch.Tensor:
        return criterion(outputs, labels)

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """Return the outer-loop meta-gradients computed during train()."""
        return self.last_gradients or {}


class FedAvgMetaTrainer(FedMetaBaseTrainer):
    """FedAvg-Meta Client Trainer."""

    def train(self, model, trainloader, epochs, lr, device) -> float:
        """Standard local SGD training on support set."""
        sup_loader, _ = split_support_query(trainloader, self.support_ratio)

        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        total_loss = 0.0
        num_batches = 0
        for _ in range(epochs):
            for batch in sup_loader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                # Skip single-sample batch to protect BatchNorm
                if images.size(0) <= 1:
                    continue

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate(self, model, valloader, device) -> Tuple[float, float]:
        """Few-shot meta-testing: 1-step Adam adaptation on support, test on query."""
        sup_loader, qry_loader = split_support_query(valloader, self.support_ratio)
        test_net = deepcopy(model).to(device)
        test_net.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(test_net.parameters(), lr=self.alpha, weight_decay=1e-3)

        for batch in sup_loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            if images.size(0) <= 1:
                continue

            optimizer.zero_grad()
            loss = criterion(test_net(images), labels)
            loss.backward()
            optimizer.step()

        # Evaluate adapted model on query set
        test_net.eval()
        return evaluate_model(test_net, qry_loader, device)


class FedMetaMamlTrainer(FedMetaBaseTrainer):
    """FedMeta-MAML Client Trainer."""

    def train(self, model, trainloader, epochs, lr, device) -> float:
        sup_loader, qry_loader = split_support_query(trainloader, self.support_ratio)

        model.to(device)
        train_net = deepcopy(model).to(device)
        train_net.train()
        criterion = nn.CrossEntropyLoss()

        step_size = self.alpha

        # 1. Inner loop adaptation on support loader
        for _ in range(self.gradient_step):
            for batch in sup_loader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                # Skip single-sample batch to protect BatchNorm
                if images.size(0) <= 1:
                    continue

                loss = criterion(train_net(images), labels)
                # create_graph=False: first-order MAML approximation.
                # We only need create_graph=True for the OUTER meta-gradient,
                # not for the inner-loop adaptation steps — this saves ~10 GB RAM.
                grads = torch.autograd.grad(
                    loss,
                    list(train_net.parameters()),
                    create_graph=False,
                    retain_graph=False,
                )

                for param, g in zip(train_net.parameters(), grads):
                    param.data = param.data - step_size * g.detach()

                for param in train_net.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

        # 2. Outer loop: per-batch gradient accumulation.
        # Old approach (accumulate loss_sum → single autograd.grad) retained
        # ALL query batch graphs simultaneously in VRAM (~9 GB for client 0).
        # New approach: loss.backward() per batch frees each graph immediately;
        # peak VRAM stays at ONE batch graph (~25 MB) at a time.
        train_net.zero_grad()
        total_loss_val = 0.0
        total_samples = 0

        for batch in qry_loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            if images.size(0) <= 1:
                continue

            n = labels.size(0)
            loss = criterion(train_net(images), labels)
            # Weight by batch size so the final gradient is a sample-weighted average
            (loss * n).backward()
            total_loss_val += loss.item() * n
            total_samples += n

        # Normalize accumulated gradients by total sample count
        if total_samples > 0:
            for param in train_net.parameters():
                if param.grad is not None:
                    param.grad /= total_samples

        # 3. Collect meta-gradients from accumulated .grad attributes
        param_names = [k for k, v in model.named_parameters() if v.requires_grad]
        self.last_gradients = {}
        for name, param in zip(param_names, train_net.parameters()):
            self.last_gradients[name] = (
                param.grad.detach().cpu() if param.grad is not None
                else torch.zeros_like(param).cpu()
            )

        for name, buf in model.named_buffers():
            self.last_gradients[name] = buf.detach().cpu()

        loss_avg = total_loss_val / max(total_samples, 1)
        return float(loss_avg)


    def evaluate(self, model, valloader, device) -> Tuple[float, float]:
        """Meta-testing adaptation on support, test on query."""
        sup_loader, qry_loader = split_support_query(valloader, self.support_ratio)
        test_net = deepcopy(model).to(device)
        test_net.train()
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.gradient_step):
            for batch in sup_loader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                if images.size(0) <= 1:
                    continue

                loss = criterion(test_net(images), labels)
                # create_graph=False: evaluation only needs adapted weights,
                # no outer meta-gradient is computed here — safe and memory-efficient.
                grads = torch.autograd.grad(
                    loss,
                    list(test_net.parameters()),
                    create_graph=False,
                    retain_graph=False,
                )
                for param, g in zip(test_net.parameters(), grads):
                    param.data = param.data - self.alpha * g.detach()

                for param in test_net.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

        test_net.eval()
        return evaluate_model(test_net, qry_loader, device)


class FedMetaMetaSgdTrainer(FedMetaMamlTrainer):
    """FedMeta-Meta-SGD Client Trainer."""

    def __init__(self, alpha_state: Optional[Dict[str, torch.Tensor]] = None, **kwargs):
        super().__init__(**kwargs)
        self.alpha_state = alpha_state

    def train(self, model, trainloader, epochs, lr, device) -> float:
        sup_loader, qry_loader = split_support_query(trainloader, self.support_ratio)

        model.to(device)
        train_net = deepcopy(model).to(device)
        train_net.train()
        criterion = nn.CrossEntropyLoss()

        param_names = [k for k, v in model.named_parameters() if v.requires_grad]
        alphas: List[torch.Tensor] = []
        for name, param in zip(param_names, train_net.parameters()):
            if self.alpha_state and name in self.alpha_state:
                alphas.append(self.alpha_state[name].to(device))
            else:
                alphas.append(torch.full_like(param, self.alpha, device=device))

        # 1. Inner loop adaptation with learnable alpha vectors
        for _ in range(self.gradient_step):
            for batch in sup_loader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                if images.size(0) <= 1:
                    continue

                loss = criterion(train_net(images), labels)
                # create_graph=False: first-order approximation for inner-loop steps.
                # The alpha vectors are updated by the outer meta-gradient on the query set,
                # not by second-order gradients through the inner loop.
                grads = torch.autograd.grad(
                    loss,
                    list(train_net.parameters()),
                    create_graph=False,
                    retain_graph=False,
                )

                for param, g, a in zip(train_net.parameters(), grads, alphas):
                    param.data = param.data - a * g.detach()

                for param in train_net.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

        # 2. Outer loop: per-batch gradient accumulation (same fix as FedMetaMamlTrainer).
        # Accumulating loss across all batches and calling autograd.grad() once
        # would hold all query batch graphs in VRAM simultaneously (~9 GB OOM).
        train_net.zero_grad()
        total_loss_val = 0.0
        total_samples = 0

        for batch in qry_loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            if images.size(0) <= 1:
                continue

            n = labels.size(0)
            loss = criterion(train_net(images), labels)
            (loss * n).backward()
            total_loss_val += loss.item() * n
            total_samples += n

        # Normalize accumulated gradients by total sample count
        if total_samples > 0:
            for param in train_net.parameters():
                if param.grad is not None:
                    param.grad /= total_samples

        # 3. Collect meta-gradients; also use same grad for alpha (Meta-SGD update)
        self.last_gradients = {}
        for name, param in zip(param_names, train_net.parameters()):
            g = param.grad.detach().cpu() if param.grad is not None else torch.zeros_like(param).cpu()
            self.last_gradients[name] = g
            self.last_gradients[f"alpha.{name}"] = g

        for name, buf in model.named_buffers():
            self.last_gradients[name] = buf.detach().cpu()

        loss_avg = total_loss_val / max(total_samples, 1)
        return float(loss_avg)