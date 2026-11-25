"""FedCM-MOON Server Strategy (Flower ServerApp).

Maintains and broadcasts:
  1. Global Model W_g: weighted average over all clients
  2. Modality Group Models W_D & W_C: weighted averages over respective modality clients
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

from utils.fedcm_moon_helper import compute_lambda_t, save_group_models


class FedCMMoon(FedAvg):
    """Federated Cross-Modal MOON Strategy with Modality Group Aggregation."""

    def __init__(
        self,
        mu: float = 1.0,
        lambda_param: float = 1.0,
        temperature: float = 0.5,
        warmup_rounds: int = 10,
        ramp_rounds: int = 10,
        num_rounds: int = 20,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mu = float(mu)
        self.lambda_param = float(lambda_param)
        self.temperature = float(temperature)
        self.warmup_rounds = int(warmup_rounds)
        self.ramp_rounds = int(ramp_rounds)
        self.num_rounds = int(num_rounds)

        # Clear stale artifacts
        output_dir = Path("results/fedcm_moon")
        if output_dir.exists():
            for ptf in output_dir.glob("*.pt"):
                ptf.unlink()

        # Group models storage
        self.group_models: Dict[str, OrderedDict] = {}
        self.client_modalities: Dict[int, str] = {}

        print(
            f"[FedCM-MOON Strategy Initialized] "
            f"Total Rounds={self.num_rounds}, Warmup T_w={self.warmup_rounds}, "
            f"Ramp T_ramp={self.ramp_rounds}, Mu={self.mu}, "
            f"Lambda={self.lambda_param}, Tau={self.temperature}"
        )

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid
    ) -> Iterable[Message]:
        """Configure the training round, calculate lambda_t, and broadcast W_g and group models."""
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

        # Initialize group models in Round 1 if not yet created
        global_state = arrays.to_torch_state_dict()
        if not self.group_models:
            # Pre-initialize default modality group models to W_g
            for default_mod in ("MRI", "CT", "Dermoscopy", "Clinical"):
                self.group_models[default_mod] = copy.deepcopy(global_state)

        # Persist group models for client loading
        save_group_models(self.group_models)

        stage_name = "WARM-UP (Exact MOON, Lambda_t=0)" if lambda_t == 0.0 else f"GROUP CONTRASTIVE (Lambda_t={lambda_t:.4f})"
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
        """Aggregate global model W_g and modality-specific group models W_grp."""
        valid_replies, error_replies = self._check_and_log_replies(replies, is_train=True, validate=True)
        if error_replies:
            raise RuntimeError(
                f"[FedCM-MOON Error] Round {server_round} encountered {len(error_replies)} client failure(s)."
            )
        if not valid_replies:
            raise RuntimeError(f"[FedCM-MOON Error] Round {server_round} received 0 valid training replies.")

        # Collect client updates and group memberships
        all_weights: List[float] = []
        all_states: List[OrderedDict] = []
        group_weights: Dict[str, List[float]] = {}
        group_states: Dict[str, List[OrderedDict]] = {}

        for msg in valid_replies:
            metrics = msg.content["metrics"]
            cid = int(metrics["partition-id"])
            weight = float(metrics[self.weighted_by_key])
            modality = str(metrics.get("modality", "Unknown"))
            self.client_modalities[cid] = modality

            state_dict = msg.content["arrays"].to_torch_state_dict()

            all_weights.append(weight)
            all_states.append(state_dict)

            if modality not in group_weights:
                group_weights[modality] = []
                group_states[modality] = []

            group_weights[modality].append(weight)
            group_states[modality].append(state_dict)

        # 1. Aggregate Global Model W_g = sum_k (n_k / n) * W_k
        global_state = self._aggregate_states(all_states, all_weights)

        # 2. Aggregate Modality Group Models W_grp = sum_{k in grp} (n_k / n_grp) * W_k
        group_summaries = []
        for mod, states in group_states.items():
            weights = group_weights[mod]
            self.group_models[mod] = self._aggregate_states(states, weights)
            group_summaries.append(f"{mod} (N={sum(weights):.0f}, Clients={len(states)})")

        save_group_models(self.group_models)

        print(
            f"\n[FedCM-MOON Server Round {server_round} Aggregation]\n"
            f"  ├── Global Model W_g: {len(all_states)} clients, {sum(all_weights):.0f} total samples\n"
            f"  └── Group Models: {', '.join(group_summaries)}"
        )

        # Compute average client training loss
        train_losses = [
            float(msg.content["metrics"]["train_loss"])
            for msg in valid_replies
            if "train_loss" in msg.content["metrics"]
        ]
        avg_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0

        return ArrayRecord(global_state), {"train_loss": avg_loss}

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
