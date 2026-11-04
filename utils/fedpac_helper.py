"""
FedPAC helper utilities: Feature Alignment and Classifier Collaboration.
Reference: Xu et al., "Personalized Federated Learning with Feature Alignment
and Classifier Collaboration", ICLR 2023.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
from utils.fedrep_helper import (
    get_fedrep_head_state_dict,
    get_fedrep_shared_state_dict,
    get_head_name,
    load_fedrep_head_state_dict,
)


def get_classifier_head(model: nn.Module, head_name: Optional[str] = None) -> OrderedDict:
    """Extract classifier head parameters from model."""
    return get_fedrep_head_state_dict(model, head_name)


def get_feature_backbone(model: nn.Module, head_name: Optional[str] = None) -> OrderedDict:
    """Extract shared backbone feature extractor parameters from model."""
    return get_fedrep_shared_state_dict(model, head_name)


def load_classifier_head(model: nn.Module, head_state: OrderedDict) -> None:
    """Load classifier head parameters into model."""
    load_fedrep_head_state_dict(model, head_state)


def compute_pairwise_prototype_distances(
    client_prototypes: Dict[int, Dict[int, torch.Tensor]],
    client_counts: Dict[int, Dict[int, int]],
    client_ids: List[int],
) -> np.ndarray:
    """Compute pairwise feature distribution discrepancy matrix D(i, j)

    D(i, j) = sum_{c} p_{i, c} * || mu_{i, c} - mu_{j, c} ||^2
    where p_{i, c} is class prior probability on client i.
    """
    K = len(client_ids)
    D = np.zeros((K, K), dtype=np.float64)

    for i_idx, i_id in enumerate(client_ids):
        p_i = client_prototypes.get(i_id, {})
        c_i = client_counts.get(i_id, {})
        total_samples_i = sum(c_i.values()) or 1.0

        for j_idx, j_id in enumerate(client_ids):
            if i_idx == j_idx:
                D[i_idx, j_idx] = 0.0
                continue

            p_j = client_prototypes.get(j_id, {})
            dist_sum = 0.0

            for class_id, proto_i in p_i.items():
                prob_i_c = c_i.get(class_id, 0) / total_samples_i
                if class_id in p_j:
                    proto_j = p_j[class_id]
                    diff = proto_i.float() - proto_j.float()
                    dist_sum += prob_i_c * torch.sum(diff ** 2).item()
                else:
                    # Penalty for missing class
                    dist_sum += prob_i_c * torch.sum(proto_i.float() ** 2).item()

            D[i_idx, j_idx] = float(dist_sum)

    return D


def solve_collaboration_weights(
    distances: np.ndarray,
    sample_sizes: np.ndarray,
    tau: float = 1.0,
) -> np.ndarray:
    """Solve the Quadratic Programming problem for client classifier collaboration weights:

    min_{alpha} 0.5 * sum_{j} (tau / n_j) * alpha_j^2 + sum_{j} D(i, j) * alpha_j
    s.t. sum(alpha) = 1, alpha_j >= 0.
    """
    K = len(distances)
    if K == 1:
        return np.array([1.0], dtype=np.float64)

    sample_sizes = np.array(sample_sizes, dtype=np.float64)
    # Avoid zero division
    sample_sizes = np.maximum(sample_sizes, 1.0)
    distances = np.array(distances, dtype=np.float64)

    # Normalize distances to prevent scaling mismatch
    max_d = np.max(distances)
    if max_d > 0:
        norm_distances = distances / max_d
    else:
        norm_distances = distances

    def objective(alpha):
        var_term = 0.5 * np.sum((tau / sample_sizes) * (alpha ** 2))
        bias_term = np.sum(norm_distances * alpha)
        return var_term + bias_term

    def grad(alpha):
        return (tau / sample_sizes) * alpha + norm_distances

    constraints = {
        "type": "eq",
        "fun": lambda a: np.sum(a) - 1.0,
        "jac": lambda a: np.ones_like(a),
    }
    bounds = [(0.0, 1.0) for _ in range(K)]
    init_alpha = np.ones(K, dtype=np.float64) / K

    result = scipy.optimize.minimize(
        objective,
        init_alpha,
        method="SLSQP",
        jac=grad,
        bounds=bounds,
        constraints=constraints,
    )

    alpha = np.clip(result.x, 0.0, 1.0)
    s = np.sum(alpha)
    if s > 0:
        alpha = alpha / s
    else:
        alpha = init_alpha

    return alpha


def collaborate_classifiers(
    client_heads: Dict[int, OrderedDict],
    collaboration_matrix: Dict[int, Dict[int, float]],
) -> Dict[int, OrderedDict]:
    """Combine client classifier heads according to collaboration weights:

    W_i^{collab} = sum_{j} alpha_{i, j} * W_j.
    """
    collaborative_heads: Dict[int, OrderedDict] = {}

    for i_id, weights_dict in collaboration_matrix.items():
        # Initialize zero state dict with structure of client heads
        template_head = next(iter(client_heads.values()))
        new_head = OrderedDict()

        for key, tensor in template_head.items():
            new_head[key] = torch.zeros_like(tensor, dtype=torch.float32)

        for j_id, alpha_ij in weights_dict.items():
            if j_id in client_heads and alpha_ij > 0:
                head_j = client_heads[j_id]
                for key in new_head:
                    new_head[key] += float(alpha_ij) * head_j[key].float()

        collaborative_heads[i_id] = new_head

    return collaborative_heads
