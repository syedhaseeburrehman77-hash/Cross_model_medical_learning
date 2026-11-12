"""Generic MOON helpers for any model ending in a Linear classifier."""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoonProjectionHead(nn.Module):
    """Projection MLP used for MOON's contrastive objective."""

    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, projection_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.l1(features))
        return self.l2(features)


def _find_feature_module(model: nn.Module) -> Tuple[str, nn.Module, int]:
    """Find the classification head entry point and its input feature dimension.
    This captures features BEFORE any classifier layers or classifier dropout,
    extracting the clean backbone representation as in the official MOON baseline.
    """
    if hasattr(model, "classifier"):
        target_name = "classifier"
        target_module = model.classifier
        if isinstance(target_module, nn.Sequential):
            for sub_module in target_module:
                if isinstance(sub_module, nn.Linear):
                    return target_name, target_module, sub_module.in_features
        elif isinstance(target_module, nn.Linear):
            return target_name, target_module, target_module.in_features

    if hasattr(model, "fc3") and isinstance(model.fc3, nn.Linear):
        return "fc3", model.fc3, model.fc3.in_features

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return "fc", model.fc, model.fc.in_features

    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not name.startswith("moon_projector.")
    ]
    if not linear_layers:
        raise ValueError("MOON requires a model containing at least one nn.Linear layer.")
    return linear_layers[0][0], linear_layers[0][1], linear_layers[0][1].in_features


def prepare_moon_model(model: nn.Module, projection_dim: int = 256) -> nn.Module:
    """Attach the MOON projection head to an existing model.
    The projection head becomes part of model.state_dict(), so Flower sends,
    aggregates, and restores it automatically.
    """
    if hasattr(model, "moon_projector"):
        return model

    feature_module_name, _, in_features = _find_feature_module(model)
    model.add_module(
        "moon_projector",
        MoonProjectionHead(
            input_dim=in_features,
            projection_dim=projection_dim,
        ),
    )
    model._moon_feature_module_name = feature_module_name
    return model


class MoonFeatureForward:
    """Captures the backbone features passed into the classifier during forward()."""

    def __init__(self, model: nn.Module):
        if not hasattr(model, "moon_projector"):
            raise ValueError(
                "Call prepare_moon_model(model, projection_dim) before creating MoonFeatureForward."
            )
        self.model = model
        self.features = None
        feature_module_name = getattr(model, "_moon_feature_module_name", None)
        if feature_module_name is not None and hasattr(model, feature_module_name):
            target_module = getattr(model, feature_module_name)
        else:
            _, target_module, _ = _find_feature_module(model)
        self.hook = target_module.register_forward_pre_hook(self._capture_features)

    def _capture_features(self, _module, inputs) -> None:
        features = inputs[0]
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        self.features = features

    def forward(self, images: torch.Tensor):
        """Return representation, projected representation, and logits."""
        self.features = None
        logits = self.model(images)
        if self.features is None:
            raise RuntimeError("MOON could not capture backbone features.")
        projection = self.model.moon_projector(self.features)
        return self.features, projection, logits

    def close(self) -> None:
        """Remove the registered PyTorch hook."""
        self.hook.remove()