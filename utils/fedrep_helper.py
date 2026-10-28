"""Helpers for separating FedRep shared and local model state."""

from collections import OrderedDict

def get_head_name(model):
    """Automatically detect whether the head is 'classifier' (MobileNet/ResNet/VGG) or 'fc3' (CNN)."""
    if hasattr(model, "classifier"):
        return "classifier"
    elif hasattr(model, "fc3"):
        return "fc3"
    elif hasattr(model, "fc"):
        return "fc"
    return "classifier"

def get_fedrep_head_state_dict(model, head_name=None):
    """Return the client-specific classification head state."""
    if head_name is None:
        head_name = get_head_name(model)
    prefix = f"{head_name}."

    return OrderedDict(
        (key, value)
        for key, value in model.state_dict().items()
        if key.startswith(prefix)
    )


def get_fedrep_shared_state_dict(model, head_name=None):
    """Return model state excluding the classification head."""
    if head_name is None:
        head_name = get_head_name(model)
    prefix = f"{head_name}."
    return OrderedDict(
        (key, value)
        for key, value in model.state_dict().items()
        if not key.startswith(prefix)
    )


def load_fedrep_head_state_dict(model, head_state):
    """Restore a client's local classification head."""
    if head_state:
        model.load_state_dict(head_state, strict=False)