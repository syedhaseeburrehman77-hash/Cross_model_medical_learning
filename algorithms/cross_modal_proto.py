"""Cross-Modal Prototype Server Strategy (Flower ServerApp)."""

from collections import OrderedDict
import copy
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import sample_nodes

from utils.cross_modal_helper import (
    compute_modality_master_prototypes,
    update_ema_prototypes,
    compute_directional_transfer_weights,
    get_opposite_modality,
    save_cross_modal_guidance,
    export_cross_modal_summary,
)


class CrossModalProto(FedAvg):
    """Cross-Modal Prototype Server Strategy (without FedBN)."""

    def __init__(
        self,
        warmup_rounds: int = 2,
        warmup_fraction: Optional[float] = None,
        align_lambda: float = 0.5,
        transfer_tau: float = 15.0,
        proto_ema_beta: float = 0.7,
        num_rounds: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_rounds = int(num_rounds)
        self.align_lambda = float(align_lambda)
        self.transfer_tau = float(transfer_tau)
        self.proto_ema_beta = float(proto_ema_beta)

        if warmup_rounds is not None:
            self.warmup_rounds = int(warmup_rounds)
        elif warmup_fraction is not None:
            self.warmup_rounds = max(1, int(self.num_rounds * float(warmup_fraction)))
        else:
            self.warmup_rounds = 2

        if not (0.0 <= self.proto_ema_beta < 1.0):
            raise ValueError(f"[CrossModalProto Error] proto_ema_beta must be in [0.0, 1.0), got {self.proto_ema_beta}.")

        # Clear stale cross-modal artifacts from previous runs
        cm_dir = Path("results/cross_modal_proto")
        if cm_dir.exists():
            for stale_file in cm_dir.glob("*.pt"):
                stale_file.unlink()
            summary_csv = cm_dir / "cross_modal_transfer_summary.csv"
            if summary_csv.exists():
                summary_csv.unlink()

        # Internal state
        self.master_prototypes: Dict[str, Dict[int, torch.Tensor]] = {}
        self.modality_counts: Dict[str, Dict[int, int]] = {}
        self.transfer_weights: Dict[int, Dict[int, float]] = {}
        self.client_modalities: Dict[int, str] = {}
        self.client_counts: Dict[int, Dict[int, int]] = {}
        self.client_prototypes: Dict[int, Dict[int, torch.Tensor]] = {}

        print(
            f"[CrossModalProto Strategy Initialized] "
            f"Total Rounds={self.num_rounds}, Warm-Up Rounds={self.warmup_rounds}, "
            f"Align Lambda={self.align_lambda}, "
            f"Transfer Tau={self.transfer_tau}, EMA Beta={self.proto_ema_beta}"
        )

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid
    ) -> Iterable[Message]:
        """Configure the training round and distribute guidance to clients."""
        all_node_ids = list(grid.get_node_ids())
        num_nodes = int(len(all_node_ids) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, _ = sample_nodes(grid, self.min_available_nodes, sample_size)

        is_warmup = server_round <= self.warmup_rounds
        phase_name = "WARM-UP PHASE" if is_warmup else "CROSS-MODAL PROTOTYPE ALIGNMENT"
        print(f"\n{'='*75}")
        print(f" [CrossModalProto Server Round {server_round}/{self.num_rounds}] Stage: {phase_name}")
        print(f"{'='*75}")

        messages = []
        for node_id in node_ids:
            client_config = copy.deepcopy(config)
            client_config["server-round"] = server_round
            client_config["is-warmup"] = is_warmup
            client_config["align-lambda"] = float(self.align_lambda)

            record = RecordDict({
                self.arrayrecord_key: arrays,
                self.configrecord_key: client_config,
            })
            messages.append(
                Message(content=record, message_type=MessageType.TRAIN, dst_node_id=node_id)
            )

        return messages

    def aggregate_train(self, server_round: int, replies):
        """Aggregate global model parameters via FedAvg and process client prototypes."""
        valid_replies, error_replies = self._check_and_log_replies(replies, is_train=True, validate=True)
        if error_replies:
            raise RuntimeError(
                f"[CrossModalProto Error] Round {server_round} encountered {len(error_replies)} client failure(s)."
            )
        if not valid_replies:
            raise RuntimeError(f"[CrossModalProto Error] Round {server_round} received 0 valid training replies.")

        weights: List[float] = []
        client_states: List[OrderedDict] = []
        proto_dir = Path("results/cross_modal_proto")

        for msg in valid_replies:
            metrics = msg.content["metrics"]
            if "partition-id" not in metrics:
                raise KeyError("[CrossModalProto Error] Missing 'partition-id' in client metrics.")
            if self.weighted_by_key not in metrics:
                raise KeyError(f"[CrossModalProto Error] Missing '{self.weighted_by_key}' in client metrics.")

            cid = int(metrics["partition-id"])
            weights.append(float(metrics[self.weighted_by_key]))
            client_states.append(msg.content["arrays"].to_torch_state_dict())

            proto_file = proto_dir / f"client_{cid}_latest_prototype.pt"
            if not proto_file.exists():
                raise FileNotFoundError(
                    f"[CrossModalProto Error] Missing prototype file for client {cid} in round {server_round}: {proto_file}"
                )

            pdata = torch.load(proto_file, map_location="cpu", weights_only=False)
            saved_round = pdata.get("round")
            if saved_round != server_round:
                raise RuntimeError(
                    f"[CrossModalProto Error] Stale prototype file for client {cid}: "
                    f"found round {saved_round}, expected round {server_round}."
                )

            self.client_prototypes[cid] = pdata["prototypes"]
            self.client_counts[cid] = pdata["counts"]
            self.client_modalities[cid] = pdata["modality"]

        total_weight = sum(weights)
        if total_weight <= 0.0:
            raise ValueError(f"[CrossModalProto Error] Total aggregation weight must be > 0, got {total_weight}.")

        # 1. Aggregate shared model weights via FedAvg
        template_state = client_states[0]
        aggregated_state = OrderedDict()
        for key, tensor in template_state.items():
            aggregated_state[key] = torch.zeros_like(tensor, dtype=torch.float32)

        for w, state in zip(weights, client_states):
            norm_w = w / total_weight
            for key in aggregated_state:
                aggregated_state[key] += norm_w * state[key].float()

        for key, template_tensor in template_state.items():
            if not template_tensor.is_floating_point():
                aggregated_state[key] = aggregated_state[key].round().to(template_tensor.dtype)

        # 2. Modality Prototype Grouping, L2 Normalization, EMA Smoothing & Directional Transfer Weights
        round_master_prototypes, self.modality_counts = compute_modality_master_prototypes(
            client_prototypes=self.client_prototypes,
            client_counts=self.client_counts,
            client_modalities=self.client_modalities,
        )

        self.master_prototypes = update_ema_prototypes(
            prev_ema_prototypes=self.master_prototypes,
            current_master_prototypes=round_master_prototypes,
            ema_beta=self.proto_ema_beta,
        )

        self.transfer_weights = compute_directional_transfer_weights(
            client_counts=self.client_counts,
            client_modalities=self.client_modalities,
            modality_counts=self.modality_counts,
            tau=self.transfer_tau,
        )

        # Save EMA-smoothed guidance for next round
        is_next_warmup = (server_round + 1) <= self.warmup_rounds
        save_cross_modal_guidance(
            server_round=server_round + 1,
            master_prototypes=self.master_prototypes,
            transfer_weights=self.transfer_weights,
            client_modalities=self.client_modalities,
            is_warmup=is_next_warmup,
        )

        export_cross_modal_summary(
            server_round=server_round,
            modality_counts=self.modality_counts,
            transfer_weights=self.transfer_weights,
            client_modalities=self.client_modalities,
        )

        self._log_server_status()

        avg_loss = float(np.mean([float(msg.content["metrics"]["train_loss"]) for msg in valid_replies]))
        return ArrayRecord(aggregated_state), {"train_loss": avg_loss}

    def _log_server_status(self):
        """Print visible information to inspect EMA Master Prototypes and Directional Transfer Weights."""
        print("\n--- [EMA-Smoothed Normalized Master Modality Prototypes] ---")
        for mod, protos in self.master_prototypes.items():
            classes_info = []
            for c, p in sorted(protos.items()):
                cnt = self.modality_counts[mod][c]
                classes_info.append(f"Class {c} (N={cnt}, norm={p.norm().item():.3f})")
            print(f"  * Master {mod} Prototype: {', '.join(classes_info)}")

        print("\n--- [Smart Tutor Directional Transfer Weights] ---")
        all_mods = set(self.master_prototypes.keys())
        for cid in sorted(self.transfer_weights.keys()):
            mod = self.client_modalities[cid]
            opp_mod = get_opposite_modality(mod, all_mods)
            if opp_mod is None:
                raise ValueError(f"[CrossModalProto Error] Missing opposite modality for client {cid} ({mod}).")
            weights = self.transfer_weights[cid]
            weights_str = [
                f"Class {c} [Local N={self.client_counts[cid].get(c, 0)}]: W={w:.4f}"
                for c, w in sorted(weights.items())
            ]
            print(f"  * Hospital {cid} ({mod} -> learns from {opp_mod}): {', '.join(weights_str)}")
        print(f"{'-'*75}\n")
