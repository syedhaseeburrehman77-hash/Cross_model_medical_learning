"""
Helper utilities for Cross-Modal Federated Learning:
Prototype Grouping, L2 Normalization, EMA Smoothing, and Directional Transfer Weights.
"""

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


def get_opposite_modality(modality: str, all_modalities: Iterable[str]) -> Optional[str]:
    """Find the opposing modality from the available modalities."""
    candidates = sorted([m for m in all_modalities if m != modality])
    return candidates[0] if candidates else None


def compute_confidence(sample_count: int, tau: float = 15.0) -> float:
    """Compute confidence score based on sample abundance: count / (count + tau)."""
    if tau <= 0.0:
        raise ValueError(f"[CrossModal Error] Transfer tau must be > 0, got {tau}.")
    if sample_count < 0:
        raise ValueError(f"[CrossModal Error] Sample count cannot be negative, got {sample_count}.")
    if sample_count == 0:
        return 0.0
    return float(sample_count) / (float(sample_count) + float(tau))


def compute_modality_master_prototypes(
    client_prototypes: Dict[int, Dict[int, torch.Tensor]],
    client_counts: Dict[int, Dict[int, int]],
    client_modalities: Dict[int, str],
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, Dict[int, int]]]:
    """Group L2-normalized client prototypes by imaging modality (e.g., 'CT' vs 'MRI')
    and aggregate them into unit-norm Master Modality Prototypes weighted by client class counts.
    """
    modality_sums: Dict[str, Dict[int, torch.Tensor]] = {}
    modality_counts: Dict[str, Dict[int, int]] = {}

    for client_id, protos in client_prototypes.items():
        if client_id not in client_modalities:
            raise ValueError(f"[CrossModal Error] Missing modality metadata for client {client_id}.")
        if client_id not in client_counts:
            raise ValueError(f"[CrossModal Error] Missing class counts for client {client_id}.")

        modality = client_modalities[client_id]
        if modality not in modality_sums:
            modality_sums[modality] = {}
            modality_counts[modality] = {}

        counts = client_counts[client_id]
        for class_id, proto in protos.items():
            count = counts.get(class_id, 0)
            if count <= 0:
                continue
            proto_vec = F.normalize(proto.detach().float(), p=2, dim=0, eps=1e-8)
            weighted_proto = proto_vec * float(count)

            if class_id not in modality_sums[modality]:
                modality_sums[modality][class_id] = weighted_proto.clone()
                modality_counts[modality][class_id] = count
            else:
                modality_sums[modality][class_id] += weighted_proto.to(modality_sums[modality][class_id].device)
                modality_counts[modality][class_id] += count

    master_prototypes: Dict[str, Dict[int, torch.Tensor]] = {}
    for mod, classes_dict in modality_sums.items():
        master_prototypes[mod] = {}
        for class_id, sum_proto in classes_dict.items():
            tot_count = modality_counts[mod][class_id]
            if tot_count > 0:
                master_prototypes[mod][class_id] = F.normalize(
                    sum_proto / float(tot_count), p=2, dim=0, eps=1e-8
                )

    if not master_prototypes:
        raise ValueError("[CrossModal Error] Failed to compute any modality master prototypes.")

    return master_prototypes, modality_counts


def update_ema_prototypes(
    prev_ema_prototypes: Dict[str, Dict[int, torch.Tensor]],
    current_master_prototypes: Dict[str, Dict[int, torch.Tensor]],
    ema_beta: float = 0.7,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Smooth modality master prototypes across rounds using Exponential Moving Average (EMA).

    Formula:
        P_ema^(t) = normalize( beta * P_ema^(t-1) + (1 - beta) * P_curr^(t) )
    """
    beta = float(ema_beta)
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"[CrossModal Error] proto_ema_beta must be in [0.0, 1.0), got {beta}.")

    smoothed: Dict[str, Dict[int, torch.Tensor]] = {}
    all_modalities = set(prev_ema_prototypes.keys()).union(current_master_prototypes.keys())

    for mod in all_modalities:
        prev_classes = prev_ema_prototypes.get(mod, {})
        curr_classes = current_master_prototypes.get(mod, {})
        all_classes = set(prev_classes.keys()).union(curr_classes.keys())

        smoothed[mod] = {}
        for class_id in all_classes:
            has_prev = class_id in prev_classes
            has_curr = class_id in curr_classes

            if has_curr and has_prev and beta > 0.0:
                prev_p = F.normalize(prev_classes[class_id].detach().float(), p=2, dim=0, eps=1e-8)
                curr_p = F.normalize(
                    curr_classes[class_id].detach().float().to(prev_p.device), p=2, dim=0, eps=1e-8
                )
                blended = (beta * prev_p) + ((1.0 - beta) * curr_p)
                smoothed[mod][class_id] = F.normalize(blended, p=2, dim=0, eps=1e-8)
            elif has_curr:
                curr_p = curr_classes[class_id].detach().float()
                smoothed[mod][class_id] = F.normalize(curr_p, p=2, dim=0, eps=1e-8)
            elif has_prev:
                prev_p = prev_classes[class_id].detach().float()
                smoothed[mod][class_id] = F.normalize(prev_p, p=2, dim=0, eps=1e-8)

    return smoothed


def compute_directional_transfer_weights(
    client_counts: Dict[int, Dict[int, int]],
    client_modalities: Dict[int, str],
    modality_counts: Dict[str, Dict[int, int]],
    tau: float = 15.0,
) -> Dict[int, Dict[int, float]]:
    """Compute directional transfer weights: Conf(source_modality) * (1 - Conf(client))."""
    transfer_weights: Dict[int, Dict[int, float]] = {}
    all_known_modalities = set(client_modalities.values()).union(modality_counts.keys())

    for client_id, counts in client_counts.items():
        rec_mod = client_modalities[client_id]
        opp_mod = get_opposite_modality(rec_mod, all_known_modalities)
        if opp_mod is None or opp_mod not in modality_counts:
            raise ValueError(
                f"[CrossModal Error] No opposite modality available for client {client_id} ({rec_mod}). "
                f"Known modalities: {sorted(all_known_modalities)}. "
                f"Ensure clients from both modalities participate in training."
            )

        transfer_weights[client_id] = {}
        all_classes = set(counts.keys()).union(modality_counts[opp_mod].keys())

        for class_id in sorted(all_classes):
            n_client = counts.get(class_id, 0)
            n_source = modality_counts[opp_mod].get(class_id, 0)

            conf_client = compute_confidence(n_client, tau=tau)
            conf_source = compute_confidence(n_source, tau=tau)
            transfer_weights[client_id][class_id] = float(conf_source * (1.0 - conf_client))

    return transfer_weights


def save_cross_modal_guidance(
    server_round: int,
    master_prototypes: Dict[str, Dict[int, torch.Tensor]],
    transfer_weights: Dict[int, Dict[int, float]],
    client_modalities: Dict[int, str],
    is_warmup: bool,
    output_dir: str = "results/cross_modal_proto",
):
    """Save EMA-smoothed opposite-modality master prototypes and transfer weights for each client."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_known_modalities = set(client_modalities.values()).union(master_prototypes.keys())

    for client_id, mod in client_modalities.items():
        opp_mod = get_opposite_modality(mod, all_known_modalities)
        if opp_mod is None or opp_mod not in master_prototypes:
            raise ValueError(
                f"[CrossModal Error] Cannot save guidance for client {client_id} ({mod}): "
                f"opposite modality prototypes are missing. Known modalities: {sorted(all_known_modalities)}."
            )

        opp_protos = {c: p.detach().cpu() for c, p in master_prototypes[opp_mod].items()}
        if not is_warmup and not opp_protos:
            raise ValueError(
                f"[CrossModal Error] Opposite modality '{opp_mod}' has empty prototypes for client {client_id}."
            )

        weights = transfer_weights[client_id]
        guidance_data = {
            "round": server_round,
            "is_warmup": is_warmup,
            "client_id": client_id,
            "modality": mod,
            "opposite_modality": opp_mod,
            "opposite_master_prototypes": opp_protos,
            "transfer_weights": weights,
        }
        torch.save(guidance_data, out_dir / f"client_{client_id}_guidance.pt")


def load_cross_modal_guidance(
    client_id: int,
    expected_round: int,
    output_dir: str = "results/cross_modal_proto",
) -> Dict[str, Any]:
    """Load guidance saved by the server for this client, raising directly on missing or stale state."""
    guidance_file = Path(output_dir) / f"client_{client_id}_guidance.pt"
    if not guidance_file.exists():
        raise FileNotFoundError(
            f"[CrossModal Error] Guidance file missing for client {client_id} at round {expected_round}: {guidance_file}"
        )

    data = torch.load(guidance_file, map_location="cpu", weights_only=False)
    saved_round = data.get("round")
    if saved_round != expected_round:
        raise RuntimeError(
            f"[CrossModal Error] Stale guidance for client {client_id}: "
            f"found round {saved_round}, expected round {expected_round}."
        )
    if not data.get("opposite_master_prototypes"):
        raise ValueError(
            f"[CrossModal Error] Empty opposite_master_prototypes in guidance for client {client_id} at round {expected_round}."
        )
    if not data.get("transfer_weights"):
        raise ValueError(
            f"[CrossModal Error] Empty transfer_weights in guidance for client {client_id} at round {expected_round}."
        )
    return data


def export_cross_modal_summary(
    server_round: int,
    modality_counts: Dict[str, Dict[int, int]],
    transfer_weights: Dict[int, Dict[int, float]],
    client_modalities: Dict[int, str],
    output_dir: str = "results/cross_modal_proto",
):
    """Save round summary to CSV for experiment tracking and analysis."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_file = out_dir / "cross_modal_transfer_summary.csv"
    file_exists = csv_file.exists()

    label_names = {0: "Healthy", 1: "Tumor"}
    all_known_modalities = set(client_modalities.values()).union(modality_counts.keys())

    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "round",
                "client_id",
                "client_modality",
                "class_id",
                "class_name",
                "transfer_weight",
                "opp_modality_count",
            ])

        for cid, weights in transfer_weights.items():
            mod = client_modalities[cid]
            opp_mod = get_opposite_modality(mod, all_known_modalities)
            if opp_mod is None or opp_mod not in modality_counts:
                raise ValueError(
                    f"[CrossModal Error] Cannot export summary for client {cid} ({mod}): missing opposite modality counts."
                )
            for cls_id, weight in weights.items():
                opp_cnt = modality_counts[opp_mod].get(cls_id, 0)
                writer.writerow([
                    server_round,
                    cid,
                    mod,
                    cls_id,
                    label_names.get(cls_id, f"Class_{cls_id}"),
                    f"{weight:.4f}",
                    opp_cnt,
                ])
