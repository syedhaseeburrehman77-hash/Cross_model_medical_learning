from collections import OrderedDict
from pathlib import Path
from typing import Dict, List
import torch
from flwr.app import ArrayRecord
from flwr.serverapp.strategy import FedAvg
from utils.fedlc_helper import is_fedlc_local_param


class FedLC(FedAvg):
    """FedLC Server Strategy: Local Calibration Federated Learning.

    Reference: Wang et al., ECCV 2022.
    Aggregates the shared backbone representations across clients via weighted averaging,
    while preserving client-specific classification heads.
    """

    def __init__(
        self,
        alpha: float = 0.0,
        pcs: bool = True,
        hc: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.pcs = bool(pcs)
        self.hc = bool(hc)
        self.client_heads: Dict[int, OrderedDict] = {}

    def aggregate_train(self, server_round: int, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True, validate=False)
        if not valid_replies:
            return None, None

        weights = [float(msg.content["metrics"].get(self.weighted_by_key, 1.0)) for msg in valid_replies]
        total_weight = sum(weights) or 1.0

        client_backbones: List[OrderedDict] = []

        for i, msg in enumerate(valid_replies):
            cid = int(msg.content["metrics"].get("partition-id", i))

            state = msg.content["arrays"].to_torch_state_dict()
            # Separate local head from shared backbone
            head_state = OrderedDict((k, v) for k, v in state.items() if is_fedlc_local_param(k))
            backbone_state = OrderedDict((k, v) for k, v in state.items() if not is_fedlc_local_param(k))

            self.client_heads[cid] = head_state
            client_backbones.append(backbone_state)

        # Aggregate shared backbone via sample-weighted federated averaging
        aggregated_backbone = OrderedDict()
        template_backbone = client_backbones[0]

        for key, tensor in template_backbone.items():
            aggregated_backbone[key] = torch.zeros_like(tensor, dtype=torch.float32)

        for w, backbone in zip(weights, client_backbones):
            norm_w = float(w) / total_weight
            for key in aggregated_backbone:
                aggregated_backbone[key] += norm_w * backbone[key].float()

        # Save heads snapshot for analysis
        save_dir = Path("results/fedlc")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"round": server_round, "client_heads": self.client_heads},
            save_dir / f"fedlc_heads_round_{server_round}.pt",
        )

        # Aggregate MetricRecords from clients
        reply_contents = [msg.content for msg in valid_replies]
        metrics = {}
        if self.train_metrics_aggr_fn:
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )

        # Construct global model state (shared backbone + averaged heads for global evaluation)
        combined_state = OrderedDict(aggregated_backbone)
        valid_heads = [h for h in self.client_heads.values() if len(h) > 0]
        if valid_heads:
            template_head = valid_heads[0]
            avg_head = OrderedDict()
            for key, tensor in template_head.items():
                avg_head[key] = torch.zeros_like(tensor, dtype=torch.float32)

            for w, (cid, head) in zip(weights, self.client_heads.items()):
                if len(head) > 0:
                    norm_w = float(w) / total_weight
                    for key in avg_head:
                        avg_head[key] += norm_w * head[key].float()

            combined_state.update(avg_head)

        return ArrayRecord(combined_state), metrics
