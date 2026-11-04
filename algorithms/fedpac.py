from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from flwr.app import ArrayRecord
from flwr.serverapp.strategy import FedAvg
from utils.fedpac_helper import (
    collaborate_classifiers,
    compute_pairwise_prototype_distances,
    solve_collaboration_weights,
)
from utils.prototype_helper import aggregate_prototypes


class FedPAC(FedAvg):
    """FedPAC Server Strategy: Feature Alignment and Classifier Collaboration.

    Reference: Xu et al., ICLR 2023.
    Aggregates backbone feature extractors via federated averaging,
    computes global class prototypes for local feature alignment, and
    collaboratively optimizes personalized classifier heads for each client.
    """

    def __init__(
        self,
        fedpac_lambda: float = 1.0,
        fedpac_tau: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fedpac_lambda = float(fedpac_lambda)
        self.fedpac_tau = float(fedpac_tau)
        self.client_heads: Dict[int, OrderedDict] = {}
        self.global_prototypes: Dict[int, torch.Tensor] = {}
        self.collaboration_matrix: Dict[int, Dict[int, float]] = {}

    def aggregate_train(self, server_round: int, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True, validate=False)
        if not valid_replies:
            return None, None

        weights = [float(msg.content["metrics"].get(self.weighted_by_key, 1.0)) for msg in valid_replies]
        total_weight = sum(weights) or 1.0

        client_ids: List[int] = []
        client_heads: Dict[int, OrderedDict] = {}
        client_backbones: List[OrderedDict] = []
        client_sample_sizes: List[int] = []

        proto_dir = Path("results/prototypes")
        client_prototypes: Dict[int, Dict[int, torch.Tensor]] = {}
        client_counts: Dict[int, Dict[int, int]] = {}

        for msg in valid_replies:
            cid = int(msg.content["metrics"].get("partition-id", len(client_ids)))
            client_ids.append(cid)
            sample_size = int(msg.content["metrics"].get(self.weighted_by_key, 1))
            client_sample_sizes.append(sample_size)

            state = msg.content["arrays"].to_torch_state_dict()
            head_state = OrderedDict((k, v) for k, v in state.items() if "classifier" in k or "fc3" in k)
            backbone_state = OrderedDict((k, v) for k, v in state.items() if not ("classifier" in k or "fc3" in k))

            client_heads[cid] = head_state
            client_backbones.append(backbone_state)

            # Load prototypes if saved by client
            proto_file = proto_dir / f"client_{cid}_latest_prototype.pt"
            if proto_file.exists():
                try:
                    pdata = torch.load(proto_file, weights_only=False)
                    client_prototypes[cid] = pdata["prototypes"]
                    client_counts[cid] = pdata["counts"]
                except Exception:
                    pass

        # 1. Aggregate shared backbone via sample-weighted averaging
        aggregated_backbone = OrderedDict()
        template_backbone = client_backbones[0]

        for key, tensor in template_backbone.items():
            aggregated_backbone[key] = torch.zeros_like(tensor, dtype=torch.float32)

        for w, backbone in zip(weights, client_backbones):
            norm_w = float(w) / total_weight
            for key in aggregated_backbone:
                aggregated_backbone[key] += norm_w * backbone[key].float()

        # 2. Aggregate global class prototypes
        if client_prototypes:
            self.global_prototypes = aggregate_prototypes(client_prototypes, client_counts)
            save_proto_dir = Path("results/fedpac")
            save_proto_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"global_prototypes": self.global_prototypes, "round": server_round},
                save_proto_dir / f"global_prototypes_round_{server_round}.pt",
            )

        # 3. Classifier Collaboration: compute optimal collaboration weights
        if len(client_ids) > 1 and client_prototypes:
            dist_matrix = compute_pairwise_prototype_distances(
                client_prototypes, client_counts, client_ids
            )
            collab_matrix: Dict[int, Dict[int, float]] = {}

            for idx, cid in enumerate(client_ids):
                distances = dist_matrix[idx]
                alpha = solve_collaboration_weights(
                    distances, np.array(client_sample_sizes), tau=self.fedpac_tau
                )
                collab_matrix[cid] = {
                    client_ids[j]: float(alpha[j]) for j in range(len(client_ids))
                }

            self.collaboration_matrix = collab_matrix

            # Combine client classifiers
            collab_heads = collaborate_classifiers(client_heads, collab_matrix)
            self.client_heads = collab_heads

            # Save personalized collaborative heads
            out_dir = Path("results/fedpac")
            out_dir.mkdir(parents=True, exist_ok=True)
            for cid, chead in collab_heads.items():
                torch.save(chead, out_dir / f"client_{cid}_collab_head.pt")

            print(f"\n[FedPAC] Round {server_round}: Classifier Collaboration Weights:")
            for cid, weights_map in collab_matrix.items():
                weights_str = ", ".join([f"Client {j}: {w:.4f}" for j, w in weights_map.items()])
                print(f"  ├── Client {cid} weights -> [{weights_str}]")
        else:
            self.client_heads = client_heads

        # 4. Construct global array record (shared backbone + weighted averaged head)
        combined_state = OrderedDict(aggregated_backbone)
        template_head = next(iter(client_heads.values()))
        avg_head = OrderedDict()

        for key, tensor in template_head.items():
            avg_head[key] = torch.zeros_like(tensor, dtype=torch.float32)

        for w, head in zip(weights, client_heads.values()):
            norm_w = float(w) / total_weight
            for key in avg_head:
                avg_head[key] += norm_w * head[key].float()

        combined_state.update(avg_head)

        metrics = {"train_loss": float(np.mean([msg.content["metrics"].get("train_loss", 0.0) for msg in valid_replies]))}
        return ArrayRecord(combined_state), metrics