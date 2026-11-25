"""
Helper utilities for Federated Cross-Modal MOON (FedCM-MOON / Group-Contrastive MOON).
Handles modality mapping, group model persistence, and linear schedule for lambda_t.
"""

from pathlib import Path
from typing import Dict, Optional
import torch


def get_client_modality(
    partition_id: int,
    num_partitions: int,
    dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> str:
    """Map client partition_id to its imaging modality / domain group."""
    if num_partitions == 6:
        return "MRI" if partition_id in (0, 1, 2) else "CT"
    elif num_partitions == 4:
        return "MRI" if partition_id in (0, 1) else "CT"
    elif num_partitions == 2:
        return "MRI" if partition_id == 0 else "CT"
    else:
        return "MRI" if partition_id < (num_partitions // 2) else "CT"


def compute_lambda_t(
    server_round: int,
    warmup_rounds: int = 10,
    ramp_rounds: int = 10,
    lambda_max: float = 1.0,
) -> float:
    """Compute the group contrastive weight lambda_t for communication round t.

    Schedule:
      - Rounds 1 to T_w: lambda_t = 0 (exact standard MOON).
      - Rounds T_w + 1 to T_w + T_ramp: ramp lambda_t linearly from 0 to lambda_max.
      - Beyond T_w + T_ramp: hold at lambda_max.
    """
    if server_round <= warmup_rounds:
        return 0.0
    if ramp_rounds <= 0:
        return float(lambda_max)

    progress = min(1.0, float(server_round - warmup_rounds) / float(ramp_rounds))
    return float(lambda_max * progress)


def save_group_models(
    group_models: Dict[str, Dict[str, torch.Tensor]],
    output_dir: str = "results/fedcm_moon",
) -> None:
    """Save group models to disk for client retrieval."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for modality, state_dict in group_models.items():
        clean_state = {k: v.detach().cpu() for k, v in state_dict.items()}
        torch.save(clean_state, out_dir / f"group_model_{modality}.pt")


def load_group_model(
    modality: str,
    input_dir: str = "results/fedcm_moon",
) -> Optional[Dict[str, torch.Tensor]]:
    """Load the group model for a client's modality.
    
    Group models are saved as Group_0 (MRI/Dermoscopy) or Group_1 (CT/Clinical)
    to avoid passing strings through Flower's MetricRecord (which only accepts int/float).
    """
    # Map semantic modality name → group file key
    _MODALITY_TO_GROUP = {
        "MRI": "Group_0",
        "Dermoscopy": "Group_0",
        "CT": "Group_1",
        "Clinical": "Group_1",
    }
    group_key = _MODALITY_TO_GROUP.get(modality, "Group_0")
    group_file = Path(input_dir) / f"group_model_{group_key}.pt"
    if not group_file.exists():
        return None
    return torch.load(group_file, map_location="cpu", weights_only=False)
