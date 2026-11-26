"""FedCM-MOON Server Strategy (Flower ServerApp).

Maintains and broadcasts:
  1. Global Model W_g: weighted average over all clients
  2. Class-Wise Modality Master Prototypes P_c^m with EMA Smoothing
  3. Directional Cross-Modal Transfer Weights g_{k,c}
Ramps lambda_t linearly from 0 to lambda after warm-up rounds.
"""

from collections import OrderedDict
import copy
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import sample_nodes

from utils.fedcm_moon_helper import compute_lambda_t
from utils.cross_modal_helper import (
    compute_modality_master_prototypes,
    update_ema_prototypes,
    compute_directional_transfer_weights,
    get_opposite_modality,
    save_cross_modal_guidance,
)


class FedCMMoon(FedAvg):
    """Federated Cross-Modal MOON Strategy with Class-Wise Prototype Aggregation."""

    def __init__(
        self,
        mu: float = 1.0,
        lambda_param: float = 0.5,
        temperature: float = 0.5,
        warmup_rounds: int = 1,
        ramp_rounds: int = 1,
        proto_ema_beta: float = 0.7,
        transfer_tau: float = 15.0,
        num_rounds: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mu = float(mu)
        self.lambda_param = float(lambda_param)
        self.temperature = float(temperature)
        self.warmup_rounds = int(warmup_rounds)
        self.ramp_rounds = int(ramp_rounds)
        self.proto_ema_beta = float(proto_ema_beta)
        self.transfer_tau = float(transfer_tau)
        self.num_rounds = int(num_rounds)

        # Clear stale artifacts
        output_dir = Path("results/fedcm_moon")
        if output_dir.exists():
            for ptf in output_dir.glob("*.pt"):
                ptf.unlink()

        # Class Prototype and Modality storage (Ideas A, B, D)
        self.master_prototypes: Dict[str, Dict[int, torch.Tensor]] = {}
        self.modality_counts: Dict[str, Dict[int, int]] = {}
        self.transfer_weights: Dict[int, Dict[int, float]] = {}
        self.client_modalities: Dict[int, str] = {}
        self.client_counts: Dict[int, Dict[int, int]] = {}
        self.client_prototypes: Dict[int, Dict[int, torch.Tensor]] = {}

        print(
            f"[FedCM-MOON Strategy Initialized] "
            f"Total Rounds={self.num_rounds}, Warmup T_w={self.warmup_rounds}, "
            f"Ramp T_ramp={self.ramp_rounds}, Mu={self.mu}, "
            f"Lambda={self.lambda_param}, Tau={self.temperature}, "
            f"EMA Beta={self.proto_ema_beta}, Transfer Tau={self.transfer_tau}"
        )

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid
    ) -> Iterable[Message]:
        """Configure the training round, calculate lambda_t, and broadcast W_g to clients."""
        all_node_ids = list(grid.get_node_ids())
        num_nodes = int(len(all_node_ids) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, _ = sample_nodes(grid, self.min_available_nodes, sample_size)

        lambda_t = compute_lambda_t(
            server_round=server_round,
            warmup_rounds=self.warmup_rounds,
            ramp_rounds=self.ramp_rounds,
            lambda_max=self.lambda_param,
        )

        stage_name = "WARM-UP (Exact MOON, Lambda_t=0)" if lambda_t == 0.0 else f"PROTOTYPE ALIGNMENT (Lambda_t={lambda_t:.4f})"
        print(f"\n{'='*75}")
        print(f" [FedCM-MOON Server Round {server_round}/{self.num_rounds}] Stage: {stage_name}")
        print(f"{'='*75}")

        messages = []
        for node_id in node_ids:
            client_config = copy.deepcopy(config)
            client_config["server-round"] = server_round
            client_config["moon-lambda-t"] = float(lambda_t)
            client_config["moon-mu"] = float(self.mu)
            client_config["moon-temperature"] = float(self.temperature)

            record = RecordDict({
                self.arrayrecord_key: arrays,
                self.configrecord_key: client_config,
            })
            messages.append(
                Message(content=record, message_type=MessageType.TRAIN, dst_node_id=node_id)
            )

        return messages

    def aggregate_train(self, server_round: int, replies):
        """Aggregate global model W_g and class-wise modality prototypes."""
        valid_replies, error_replies = self._check_and_log_replies(replies, is_train=True, validate=True)
        if error_replies:
            raise RuntimeError(
                f"[FedCM-MOON Error] Round {server_round} encountered {len(error_replies)} client failure(s)."
            )
        if not valid_replies:
            raise RuntimeError(f"[FedCM-MOON Error] Round {server_round} received 0 valid training replies.")

        all_weights: List[float] = []
        all_states: List[OrderedDict] = []
        proto_dir = Path("results/fedcm_moon")

        for msg in valid_replies:
            metrics = msg.content["metrics"]
            cid = int(metrics["partition-id"])
            weight = float(metrics[self.weighted_by_key])
            state_dict = msg.content["arrays"].to_torch_state_dict()

            all_weights.append(weight)
            all_states.append(state_dict)

            # Read client class prototypes if available
            proto_file = proto_dir / f"client_{cid}_latest_prototype.pt"
            if proto_file.exists():
                try:
                    pdata = torch.load(proto_file, map_location="cpu", weights_only=False)
                    self.client_prototypes[cid] = pdata["prototypes"]
                    self.client_counts[cid] = pdata["counts"]
                    self.client_modalities[cid] = pdata.get("modality", "MRI" if cid in (0, 1) else "CT")
                except Exception as e:
                    print(f"[FedCM-MOON Warning] Failed to read prototype for client {cid}: {e}")

        # 1. Aggregate Global Model W_g = sum_k (n_k / n) * W_k
        global_state = self._aggregate_states(all_states, all_weights)

        # 2. Modality Prototype Grouping, L2 Normalization, EMA Smoothing & Directional Transfer Weights
        if self.client_prototypes and self.client_counts:
            # Idea A: Class-Wise Modality Prototype Aggregation
            round_master_prototypes, self.modality_counts = compute_modality_master_prototypes(
                client_prototypes=self.client_prototypes,
                client_counts=self.client_counts,
                client_modalities=self.client_modalities,
            )

            # Idea D: Exponential Moving Average (EMA) Prototype Momentum (gamma = 0.7)
            self.master_prototypes = update_ema_prototypes(
                prev_ema_prototypes=self.master_prototypes,
                current_master_prototypes=round_master_prototypes,
                ema_beta=self.proto_ema_beta,
            )

            # Idea B: Directional Cross-Modal Transfer Weighting (g_{k,c})
            self.transfer_weights = compute_directional_transfer_weights(
                client_counts=self.client_counts,
                client_modalities=self.client_modalities,
                modality_counts=self.modality_counts,
                tau=self.transfer_tau,
            )

            # Save guidance for next round
            is_next_warmup = (server_round + 1) <= self.warmup_rounds
            save_cross_modal_guidance(
                server_round=server_round + 1,
                master_prototypes=self.master_prototypes,
                transfer_weights=self.transfer_weights,
                client_modalities=self.client_modalities,
                is_warmup=is_next_warmup,
                output_dir="results/fedcm_moon",
            )
            self._log_server_status()

        print(
            f"\n[FedCM-MOON Server Round {server_round} Aggregation]\n"
            f"  └── Global Model W_g: {len(all_states)} clients, {sum(all_weights):.0f} total samples"
        )

        # Compute average client training loss
        train_losses = [
            float(msg.content["metrics"]["train_loss"])
            for msg in valid_replies
            if "train_loss" in msg.content["metrics"]
        ]
        avg_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0

        return ArrayRecord(global_state), {"train_loss": avg_loss}

    def _log_server_status(self):
        """Print visible information about EMA Master Prototypes and Transfer Weights."""
        print("\n--- [FedCM-MOON EMA Master Modality Prototypes] ---")
        for mod, protos in self.master_prototypes.items():
            classes_info = [
                f"Class {c} (N={self.modality_counts[mod].get(c, 0)}, norm={p.norm().item():.3f})"
                for c, p in sorted(protos.items())
            ]
            print(f"  * Master {mod} Prototype: {', '.join(classes_info)}")

        print("\n--- [FedCM-MOON Directional Transfer Weights] ---")
        all_mods = set(self.master_prototypes.keys())
        for cid in sorted(self.transfer_weights.keys()):
            mod = self.client_modalities.get(cid, "Unknown")
            opp_mod = get_opposite_modality(mod, all_mods)
            weights = self.transfer_weights[cid]
            weights_str = [
                f"Class {c} [Local N={self.client_counts.get(cid, {}).get(c, 0)}]: W={w:.4f}"
                for c, w in sorted(weights.items())
            ]
            print(f"  * Client {cid} ({mod} -> {opp_mod}): {', '.join(weights_str)}")
        print(f"{'-'*75}\n")

    @staticmethod
    def _aggregate_states(states: List[OrderedDict], weights: List[float]) -> OrderedDict:
        """Weighted average of model parameters and BN running mean/var buffers."""
        total_weight = sum(weights)
        if total_weight <= 0.0:
            raise ValueError("[FedCM-MOON Error] Total aggregation weight must be > 0.")

        norm_weights = [w / total_weight for w in weights]
        aggregated = OrderedDict()

        first_state = states[0]
        for key, tensor in first_state.items():
            if torch.is_floating_point(tensor) or torch.is_complex(tensor):
                acc = torch.zeros_like(tensor, dtype=torch.float32)
                for state, nw in zip(states, norm_weights):
                    acc += nw * state[key].detach().to(tensor.device, dtype=torch.float32)
                aggregated[key] = acc.to(tensor.dtype)
            else:
                # Integer buffers (e.g. num_batches_tracked)
                acc_val = sum(nw * state[key].detach().float().item() for state, nw in zip(states, norm_weights))
                aggregated[key] = torch.tensor(round(acc_val), dtype=tensor.dtype, device=tensor.device)

        return aggregated
