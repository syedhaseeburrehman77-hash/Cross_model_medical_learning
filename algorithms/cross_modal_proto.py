"""Cross-Modal Prototype Server Strategy (Flower ServerApp)."""

from collections import OrderedDict
import copy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import sample_nodes

from utils.cross_modal_helper import (
    compute_modality_master_prototypes,
    compute_directional_transfer_weights,
    save_cross_modal_guidance,
    export_cross_modal_summary,
)


class CrossModalProto(FedAvg):
    """Cross-Modal Prototype Server Strategy."""

    def __init__(
        self,
        warmup_fraction: float = 0.10,
        warmup_rounds: Optional[int] = None,
        align_lambda: float = 0.5,
        transfer_tau: float = 15.0,
        num_rounds: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.warmup_fraction = float(warmup_fraction if warmup_fraction is not None else 0.10)
        self.align_lambda = float(align_lambda if align_lambda is not None else 0.5)
        self.transfer_tau = float(transfer_tau if transfer_tau is not None else 15.0)
        self.num_rounds = int(num_rounds if num_rounds is not None else 10)

        if warmup_rounds is not None:
            self.warmup_rounds = int(warmup_rounds)
        else:
            self.warmup_rounds = max(1, int(self.num_rounds * self.warmup_fraction))

        # Internal state
        self.master_prototypes: Dict[str, Dict[int, torch.Tensor]] = {}
        self.modality_counts: Dict[str, Dict[int, int]] = {}
        self.transfer_weights: Dict[int, Dict[int, float]] = {}
        self.client_modalities: Dict[int, str] = {}
        self.client_counts: Dict[int, Dict[int, int]] = {}
        self.client_prototypes: Dict[int, Dict[int, torch.Tensor]] = {}

        print(
            f"[CrossModalProto Strategy Initialized] "
            f"Total Rounds={self.num_rounds}, Warm-Up Rounds={self.warmup_rounds} "
            f"({self.warmup_fraction * 100:.0f}%), Align Lambda={self.align_lambda}, "
            f"Transfer Tau={self.transfer_tau}"
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
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True, validate=False)
        if not valid_replies:
            return None, None

        weights = [float(msg.content["metrics"].get(self.weighted_by_key, 1.0)) for msg in valid_replies]
        total_weight = sum(weights) or 1.0

        client_states: List[OrderedDict] = []
        client_ids: List[int] = []

        proto_dirs = [Path("results/cross_modal_proto"), Path("results/prototypes")]
        for msg in valid_replies:
            cid = int(msg.content["metrics"].get("partition-id", len(client_ids)))
            client_ids.append(cid)
            state = msg.content["arrays"].to_torch_state_dict()
            client_states.append(state)

            # Read client's latest prototypes and sample counts
            for pdir in proto_dirs:
                proto_file = pdir / f"client_{cid}_latest_prototype.pt"
                if proto_file.exists():
                    try:
                        pdata = torch.load(proto_file, weights_only=False)
                        self.client_prototypes[cid] = pdata["prototypes"]
                        self.client_counts[cid] = pdata["counts"]
                        self.client_modalities[cid] = pdata["modality"]
                        break
                    except Exception:
                        pass

        # 1. Aggregate shared model weights via FedAvg
        template_state = client_states[0]
        aggregated_state = OrderedDict()
        for key, tensor in template_state.items():
            aggregated_state[key] = torch.zeros_like(tensor, dtype=torch.float32)

        for w, state in zip(weights, client_states):
            norm_w = float(w) / total_weight
            for key in aggregated_state:
                aggregated_state[key] += norm_w * state[key].float()

        # 2. Modality Prototype Grouping & Smart Tutor Weight Calculation
        if self.client_prototypes:
            self.master_prototypes, self.modality_counts = compute_modality_master_prototypes(
                client_prototypes=self.client_prototypes,
                client_counts=self.client_counts,
                client_modalities=self.client_modalities,
            )

            self.transfer_weights = compute_directional_transfer_weights(
                client_counts=self.client_counts,
                client_modalities=self.client_modalities,
                modality_counts=self.modality_counts,
                tau=self.transfer_tau,
            )

            # Save round summary & guidance for next round
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
                master_prototypes=self.master_prototypes,
                modality_counts=self.modality_counts,
                transfer_weights=self.transfer_weights,
                client_modalities=self.client_modalities,
            )

            # Print detailed server-side progress
            self._log_server_status(server_round)

        avg_loss = float(np.mean([msg.content["metrics"].get("train_loss", 0.0) for msg in valid_replies]))
        metrics = {"train_loss": avg_loss}
        return ArrayRecord(aggregated_state), metrics

    def _log_server_status(self, server_round: int):
        """Print visible information to inspect Master Prototypes and Directional Transfer Weights."""
        print("\n--- [Step 5: Master Modality Prototypes] ---")
        for mod, protos in self.master_prototypes.items():
            classes_info = []
            for c, p in protos.items():
                cnt = self.modality_counts.get(mod, {}).get(c, 0)
                classes_info.append(f"Class {c} (N={cnt}, norm={p.norm().item():.3f})")
            print(f"  * Master {mod} Prototype: {', '.join(classes_info)}")

        print("\n--- [Step 6 & 7: Smart Tutor Directional Transfer Weights] ---")
        all_mods = set(self.master_prototypes.keys())
        for cid in sorted(self.transfer_weights.keys()):
            mod = self.client_modalities[cid]
            opp_candidates = [m for m in all_mods if m != mod]
            opp_mod = opp_candidates[0] if opp_candidates else ("CT" if mod == "MRI" else "MRI")
            weights = self.transfer_weights[cid]
            weights_str = []
            for c, w in sorted(weights.items()):
                c_name = f"Class {c}"
                local_n = self.client_counts.get(cid, {}).get(c, 0)
                weights_str.append(f"{c_name} [Local N={local_n}]: W={w:.4f}")
            print(f"  * Hospital {cid} ({mod} -> learns from {opp_mod}): {', '.join(weights_str)}")
        print(f"{'-'*75}\n")
