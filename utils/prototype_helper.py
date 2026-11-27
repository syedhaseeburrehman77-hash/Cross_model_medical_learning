"""
Helper utilities for Class Prototype extraction, aggregation, and analysis in Federated Learning.
"""

from typing import Dict, Tuple, Optional, Any
import csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


def find_last_linear(model: nn.Module) -> Tuple[str, nn.Linear]:
    """Find the final nn.Linear layer across any model architecture (excluding auxiliary projection heads)."""
    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not name.startswith("moon_projector")
    ]
    if not linear_layers:
        raise ValueError("Model contains no nn.Linear layers for feature extraction.")
    return linear_layers[-1]


def find_feature_capture_layer(model: nn.Module) -> Tuple[str, nn.Module]:
    """Find the layer to hook for clean feature representation capture.
    If the final classification linear layer is immediately preceded by a Dropout layer
    (e.g. nn.Dropout(0.3) in MobileNet/ResNet classifiers), hooks the Dropout layer's
    pre-hook to capture clean features BEFORE dropout is applied.
    This eliminates the train/eval dropout mismatch while preserving
    dropout regularization for the classification head.
    """
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        linear_indices = [
            i for i, mod in enumerate(model.classifier)
            if isinstance(mod, nn.Linear) and not getattr(mod, "_is_projector", False)
        ]
        if linear_indices:
            last_idx = linear_indices[-1]
            if last_idx > 0 and isinstance(model.classifier[last_idx - 1], (nn.Dropout, nn.Dropout2d, nn.Dropout1d)):
                return f"classifier.{last_idx - 1}", model.classifier[last_idx - 1]
            return f"classifier.{last_idx}", model.classifier[last_idx]

    return find_last_linear(model)


class FeatureExtractor:
    """Hooks into the model before the classification head to capture clean
    feature representations (embeddings) during forward pass, immune to dropout mismatch.
    """

    def __init__(self, model: nn.Module, before_dropout: bool = False):
        self.model = model
        self.features = None
        # Keep before_dropout=False by default so FedProto/FedPAC stay identical; fedcm_moon sets True to avoid train/eval dropout mismatch.
        layer_name, target_layer = find_feature_capture_layer(model) if before_dropout else find_last_linear(model)
        self.layer_name = layer_name
        self.hook = target_layer.register_forward_pre_hook(self._capture_hook)

    def _capture_hook(self, _module, inputs):
        feat = inputs[0]
        if feat.ndim > 2:
            feat = torch.flatten(feat, start_dim=1)
        self.features = feat

    def remove(self):
        """Remove the forward pre-hook."""
        if hasattr(self, "hook") and self.hook is not None:
            self.hook.remove()
            self.hook = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


def compute_class_prototypes(
    model: nn.Module,
    dataloader,
    device: torch.device,
    before_dropout: bool = False,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
    """Compute the mean feature representation (prototype) for each class

    according to the client's local dataset distribution.

    Args:
        model: PyTorch model.
        dataloader: DataLoader for the client's dataset (train or val).
        device: Device to run inference on.
        before_dropout: Capture features before final Dropout (used by fedcm_moon).

    Returns:
        prototypes: Dict mapping class_id -> mean feature tensor (1D, CPU).
        counts: Dict mapping class_id -> number of samples in that class.
    """
    model.eval()
    model.to(device)

    feature_sums: Dict[int, torch.Tensor] = {}
    class_counts: Dict[int, int] = {}

    with FeatureExtractor(model, before_dropout=before_dropout) as extractor:
        with torch.no_grad():
            for batch in dataloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                # Forward pass triggers the hook
                _ = model(images)
                batch_features = extractor.features

                if batch_features is None:
                    raise RuntimeError("FeatureExtractor failed to capture features during prototype computation.")

                for label_val in labels.unique().tolist():
                    label = int(label_val)
                    mask = (labels == label)
                    count = int(mask.sum().item())
                    feat_sum = batch_features[mask].detach().sum(dim=0)

                    if label not in feature_sums:
                        feature_sums[label] = feat_sum
                        class_counts[label] = count
                    else:
                        feature_sums[label] += feat_sum
                        class_counts[label] += count

    prototypes: Dict[int, torch.Tensor] = {}
    for label, total_feat in feature_sums.items():
        count = class_counts[label]
        if count > 0:
            prototypes[label] = (total_feat / float(count)).cpu()

    return prototypes, class_counts


def aggregate_prototypes(
    client_prototypes: Dict[int, Dict[int, torch.Tensor]],
    client_counts: Dict[int, Dict[int, int]],
) -> Dict[int, torch.Tensor]:
    """Aggregate per-client class prototypes into global class prototypes

    weighted by the number of samples each client has for each class.

    Args:
        client_prototypes: {client_id: {class_id: prototype_tensor}}
        client_counts: {client_id: {class_id: count}}

    Returns:
        global_prototypes: {class_id: global_prototype_tensor}
    """
    global_sums: Dict[int, torch.Tensor] = {}
    global_counts: Dict[int, int] = {}

    for client_id, protos in client_prototypes.items():
        counts = client_counts.get(client_id, {})
        for class_id, proto in protos.items():
            count = counts.get(class_id, 1)
            weighted_proto = proto * count

            if class_id not in global_sums:
                global_sums[class_id] = weighted_proto.clone()
                global_counts[class_id] = count
            else:
                global_sums[class_id] += weighted_proto
                global_counts[class_id] += count

    global_prototypes: Dict[int, torch.Tensor] = {}
    for class_id, total in global_sums.items():
        total_count = global_counts[class_id]
        if total_count > 0:
            global_prototypes[class_id] = total / total_count

    return global_prototypes


def compute_prototype_similarities(
    client_prototypes: Dict[int, Dict[int, torch.Tensor]],
    client_modalities: Optional[Dict[int, str]] = None,
) -> Dict[str, float]:
    """Compute cosine similarities between client class prototypes.

    Calculates:
      1. Cross-client same-class similarity (e.g. Client 0 Tumor vs Client 1 Tumor)
      2. Intra-client inter-class similarity (e.g. Client 0 Healthy vs Client 0 Tumor)
    """
    similarities: Dict[str, float] = {}
    client_ids = sorted(list(client_prototypes.keys()))

    # Intra-client inter-class similarity
    for cid in client_ids:
        protos = client_prototypes[cid]
        mod = f" ({client_modalities[cid]})" if client_modalities and cid in client_modalities else ""
        classes = sorted(list(protos.keys()))
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                c1, c2 = classes[i], classes[j]
                sim = F.cosine_similarity(
                    protos[c1].unsqueeze(0), protos[c2].unsqueeze(0)
                ).item()
                similarities[f"Client_{cid}{mod}_Class_{c1}_vs_Class_{c2}"] = sim

    # Cross-client same-class similarity (modality alignment)
    for i in range(len(client_ids)):
        for j in range(i + 1, len(client_ids)):
            c1_id, c2_id = client_ids[i], client_ids[j]
            mod1 = f"_{client_modalities[c1_id]}" if client_modalities and c1_id in client_modalities else ""
            mod2 = f"_{client_modalities[c2_id]}" if client_modalities and c2_id in client_modalities else ""

            p1_dict = client_prototypes[c1_id]
            p2_dict = client_prototypes[c2_id]

            common_classes = set(p1_dict.keys()).intersection(set(p2_dict.keys()))
            for cls in sorted(list(common_classes)):
                sim = F.cosine_similarity(
                    p1_dict[cls].unsqueeze(0), p2_dict[cls].unsqueeze(0)
                ).item()
                similarities[f"CrossClient_C{c1_id}{mod1}_vs_C{c2_id}{mod2}_Class_{cls}"] = sim

    return similarities


def export_prototypes_summary(
    client_prototypes: Dict[int, Dict[int, torch.Tensor]],
    client_counts: Dict[int, Dict[int, int]],
    round_num: int,
    output_dir: str = "results",
    client_modalities: Optional[Dict[int, str]] = None,
):
    """Save prototypes to a PyTorch file and append a summary to CSV."""
    out_path = Path(output_dir) / "prototypes"
    out_path.mkdir(parents=True, exist_ok=True)

    # Save raw tensors
    torch.save(
        {
            "round": round_num,
            "prototypes": client_prototypes,
            "counts": client_counts,
            "modalities": client_modalities,
        },
        out_path / f"prototypes_round_{round_num}.pt",
    )

    # Write / append to CSV
    csv_file = out_path / "prototype_summary.csv"
    file_exists = csv_file.exists()

    label_names = {0: "Healthy", 1: "Tumor"}

    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "round",
                "client_id",
                "modality",
                "class_id",
                "class_name",
                "sample_count",
                "prototype_dim",
                "prototype_norm",
            ])

        for cid, protos in client_prototypes.items():
            mod = client_modalities[cid] if client_modalities and cid in client_modalities else "MRI"
            counts = client_counts.get(cid, {})
            for cls_id, proto in protos.items():
                writer.writerow([
                    round_num,
                    cid,
                    mod,
                    cls_id,
                    label_names.get(cls_id, str(cls_id)),
                    counts.get(cls_id, 0),
                    proto.shape[0],
                    f"{proto.norm().item():.4f}",
                ])


def aggregate_and_export_round_prototypes(
    round_num: int,
    num_partitions: int,
    output_dir: str = "results",
) -> Tuple[Dict[int, torch.Tensor], Dict[str, float]]:
    """Collects saved client prototypes for the round, aggregates them,
    computes cross-client and intra-client similarities, and exports summary files.
    """
    proto_dir = Path(output_dir) / "prototypes"
    client_prototypes: Dict[int, Dict[int, torch.Tensor]] = {}
    client_counts: Dict[int, Dict[int, int]] = {}
    client_modalities: Dict[int, str] = {}

    for client_id in range(num_partitions):
        cand_files = [
            proto_dir / f"client_{client_id}_round_{round_num}.pt",
            proto_dir / f"client_{client_id}_latest_prototype.pt",
        ]
        for f in cand_files:
            if f.exists():
                try:
                    data = torch.load(f, weights_only=False)
                    client_prototypes[client_id] = data["prototypes"]
                    client_counts[client_id] = data["counts"]
                    if "modality" in data:
                        client_modalities[client_id] = data["modality"]
                    break
                except Exception:
                    pass

    if not client_prototypes:
        return {}, {}

    # Aggregate global prototypes
    global_protos = aggregate_prototypes(client_prototypes, client_counts)

    # Compute similarities
    similarities = compute_prototype_similarities(client_prototypes, client_modalities)

    # Export summary
    export_prototypes_summary(
        client_prototypes,
        client_counts,
        round_num=round_num,
        output_dir=output_dir,
        client_modalities=client_modalities,
    )

    # Save global prototype tensor
    torch.save(
        {
            "round": round_num,
            "global_prototypes": global_protos,
            "similarities": similarities,
        },
        proto_dir / f"global_prototypes_round_{round_num}.pt",
    )

    return global_protos, similarities
