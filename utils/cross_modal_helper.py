"""
Helper utilities for Cross-Modal Federated Learning:
Prototype Grouping and Directional Transfer Weights.
"""

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch


def get_opposite_modality(modality: str, all_modalities: Iterable[str]) -> Optional[str]:
    """Find the opposing modality from the available modalities."""
    candidates = sorted([m for m in all_modalities if m != modality])
    return candidates[0] if candidates else None


def compute_confidence(sample_count: int, tau: float = 15.0) -> float:
    """Compute confidence score based on sample abundance.

    Formula:
        Confidence = sample_count / (sample_count + tau)

    Properties:
        - When sample_count == 0, confidence = 0.0 (high need for guidance)
        - As sample_count increases, confidence saturates toward 1.0 (self-sufficient)
    """
    if sample_count <= 0:
        return 0.0
    return float(sample_count) / (float(sample_count) + float(tau))


def compute_modality_master_prototypes(
    client_prototypes: Dict[int, Dict[int, torch.Tensor]],
    client_counts: Dict[int, Dict[int, int]],
    client_modalities: Dict[int, str],
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, Dict[int, int]]]:
    """Group client prototypes by imaging modality (e.g., 'CT' vs 'MRI')
    and aggregate them into Master Modality Prototypes weighted by client class counts.

    Returns:
        master_prototypes: {modality: {class_id: master_prototype_tensor}}
        modality_counts: {modality: {class_id: total_sample_count}}
    """
    modality_sums: Dict[str, Dict[int, torch.Tensor]] = {}
    modality_counts: Dict[str, Dict[int, int]] = {}

    for client_id, protos in client_prototypes.items():
        modality = client_modalities[client_id]
        if modality not in modality_sums:
            modality_sums[modality] = {}
            modality_counts[modality] = {}

        counts = client_counts.get(client_id, {})
        for class_id, proto in protos.items():
            count = counts.get(class_id, 0)
            if count <= 0:
                continue
            weighted_proto = proto.detach().cpu().float() * float(count)

            if class_id not in modality_sums[modality]:
                modality_sums[modality][class_id] = weighted_proto.clone()
                modality_counts[modality][class_id] = count
            else:
                modality_sums[modality][class_id] += weighted_proto
                modality_counts[modality][class_id] += count

    master_prototypes: Dict[str, Dict[int, torch.Tensor]] = {}
    for mod, classes_dict in modality_sums.items():
        master_prototypes[mod] = {}
        for class_id, sum_proto in classes_dict.items():
            tot_count = modality_counts[mod][class_id]
            if tot_count > 0:
                master_prototypes[mod][class_id] = sum_proto / float(tot_count)

    return master_prototypes, modality_counts


def compute_directional_transfer_weights(
    client_counts: Dict[int, Dict[int, int]],
    client_modalities: Dict[int, str],
    modality_counts: Dict[str, Dict[int, int]],
    tau: float = 15.0,
) -> Dict[int, Dict[int, float]]:
    """Compute the 'Smart Tutor' directional transfer weight for each client and class.

    Formula:
        Transfer Weight = (Confidence of Source Modality) * (1 - Confidence of Receiving Client)

    Intuition:
        - If receiving client has very few samples (Confidence_client ~ 0), need is high (1 - Conf ~ 1).
        - If source modality has abundant samples (Confidence_source ~ 1), tutor quality is high.
        - Resulting weight is heavy. If receiving client already has ample samples, weight approaches 0.

    Returns:
        transfer_weights: {client_id: {class_id: float}}
    """
    transfer_weights: Dict[int, Dict[int, float]] = {}

    # Collect all known modalities in the federation
    all_known_modalities = set(client_modalities.values()).union(modality_counts.keys())

    for client_id, counts in client_counts.items():
        rec_mod = client_modalities[client_id]
        # Identify opposite modality explicitly and deterministically
        opp_mod = get_opposite_modality(rec_mod, all_known_modalities)
        transfer_weights[client_id] = {}

        # Collect all relevant classes (classes present in either this client or the opposite modality)
        all_classes = set(counts.keys())
        if opp_mod and opp_mod in modality_counts:
            all_classes = all_classes.union(modality_counts[opp_mod].keys())

        for class_id in sorted(list(all_classes)):
            n_client = counts.get(class_id, 0)
            n_source = 0
            if opp_mod and opp_mod in modality_counts:
                n_source = modality_counts[opp_mod].get(class_id, 0)

            conf_client = compute_confidence(n_client, tau=tau)
            conf_source = compute_confidence(n_source, tau=tau)

            # Smart tutor rule:
            weight = conf_source * (1.0 - conf_client)
            transfer_weights[client_id][class_id] = float(weight)

    return transfer_weights


def save_cross_modal_guidance(
    server_round: int,
    master_prototypes: Dict[str, Dict[int, torch.Tensor]],
    transfer_weights: Dict[int, Dict[int, float]],
    client_modalities: Dict[int, str],
    is_warmup: bool,
    output_dir: str = "results/cross_modal_proto",
):
    """Save guidance information for each client to disk so clients can access
    the opposite modality's master prototypes and directional transfer weights.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_known_modalities = set(client_modalities.values()).union(master_prototypes.keys())

    for client_id, mod in client_modalities.items():
        opp_mod = get_opposite_modality(mod, all_known_modalities)
        opp_protos = master_prototypes.get(opp_mod, {}) if opp_mod else {}
        weights = transfer_weights.get(client_id, {})

        guidance_data = {
            "round": server_round,
            "is_warmup": is_warmup,
            "client_id": client_id,
            "modality": mod,
            "opposite_modality": opp_mod,
            "opposite_master_prototypes": opp_protos,
            "transfer_weights": weights,
        }

        target_file = out_dir / f"client_{client_id}_guidance.pt"
        torch.save(guidance_data, target_file)


def load_cross_modal_guidance(
    client_id: int,
    expected_round: Optional[int] = None,
    output_dir: str = "results/cross_modal_proto",
) -> Optional[Dict[str, Any]]:
    """Load guidance saved by the server for this client, validating the round number."""
    guidance_file = Path(output_dir) / f"client_{client_id}_guidance.pt"
    if guidance_file.exists():
        try:
            data = torch.load(guidance_file, weights_only=False)
            if expected_round is not None:
                saved_round = data.get("round")
                if saved_round != expected_round:
                    print(
                        f"[CrossModal Warning] Stale guidance for client {client_id}: "
                        f"found round {saved_round}, expected round {expected_round}. Ignoring stale guidance."
                    )
                    return None
            return data
        except Exception as e:
            print(f"[CrossModal Warning] Failed to load guidance for client {client_id}: {e}")
            return None
    return None


def export_cross_modal_summary(
    server_round: int,
    master_prototypes: Dict[str, Dict[int, torch.Tensor]],
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
            for cls_id, weight in weights.items():
                opp_cnt = 0
                if opp_mod and opp_mod in modality_counts:
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
