"""Helpers for FedLC (Local Calibration for Federated Learning).

Provides state-dict splitting, head persistence, and calibration utilities
for classification architectures (MobileNet, ResNet, VGG, CNN, ResNet18MLB).
"""

from collections import OrderedDict
from typing import Dict, List
import torch
import torch.nn as nn


def get_head_name(model: nn.Module) -> str:
    """Identify the head module name across supported architectures."""
    if hasattr(model, "classifier"):
        return "classifier"
    elif hasattr(model, "fc3"):
        return "fc3"
    elif hasattr(model, "fc"):
        return "fc"
    return "classifier"


def is_fedlc_local_param(param_name: str) -> bool:
    """Check if a parameter belongs to the client-personalized classification head.

    FedLC keeps the final classification head local to each client and
    aggregates the shared backbone (feature extractor) across clients.

    Local (client-specific) layers:
      - 'classifier.*'  — MobileNet, ResNet, VGG classifier heads
      - 'fc3.*'         — CNN final classification layer
      - 'fc.*'          — generic final fc head (models using 'fc' as head name)

    Uses prefix matching to avoid false positives on intermediate layers
    (e.g. CNN's fc1/fc2 backbone layers must NOT be treated as local).
    """
    local_prefixes = ("classifier.", "fc3.", "fc.")
    return any(param_name.startswith(prefix) for prefix in local_prefixes)


def get_fedlc_head_state_dict(model: nn.Module) -> OrderedDict:
    """Return only the client-specific classification head parameters."""
    return OrderedDict(
        (key, value.cpu().clone())
        for key, value in model.state_dict().items()
        if is_fedlc_local_param(key)
    )


def get_fedlc_shared_state_dict(model: nn.Module) -> OrderedDict:
    """Return model state containing only shared backbone parameters."""
    return OrderedDict(
        (key, value)
        for key, value in model.state_dict().items()
        if not is_fedlc_local_param(key)
    )


def load_fedlc_head_state_dict(model: nn.Module, head_state: Dict[str, torch.Tensor]) -> None:
    """Restore client's personalized classification head parameters."""
    if head_state:
        model.load_state_dict(head_state, strict=False)


def compute_disagreement_map(predictions: List[torch.Tensor]) -> torch.Tensor:
    """Compute the disagreement map across predictions from multiple site heads."""
    if not predictions:
        raise ValueError("predictions list is empty")
    stacked = torch.stack(predictions, dim=0)
    return torch.std(stacked, dim=0)
