import wandb
from flwr.serverapp.strategy import FedAvg, FedProx, FedAvgM, FedAdam, FedAdagrad, FedYogi, FedMedian
from .fednova import FedNova
from .fedmeta import FedMeta
from .floco import Floco 
from .scaffold import Scaffold
from .fedpac import FedPAC
from .fedlc import FedLC
from .cross_modal_proto import CrossModalProto
from .fedcm_moon import FedCMMoon

STRATEGY_REGISTRY = {
    "fedavg": FedAvg,
    "fedprox": FedProx,
    "fedavgm": FedAvgM,
    "fedadam": FedAdam,
    "fedadagrad": FedAdagrad,
    "fedyogi": FedYogi,
    "fedmedian": FedMedian,
    "fednova": FedNova,
    "fedbn": FedAvg,
    "fedpara": FedAvg,
    "fedrep": FedAvg,
    "moon": FedAvg,
    "fedmlb": FedAvg,
    "fedper": FedAvg,
    "floco": Floco,
    "scaffold": Scaffold,
    # --- FedMeta Algorithms ---
    "fedavg_meta": FedAvg,      
    "fedmeta_maml": FedMeta,    
    "fedmeta_meta_sgd": FedMeta,
    # --- Prototype-based FL ---
    "fedproto": FedAvg,
    "fedpac": FedPAC,
    "fedlc": FedLC,
    "fedcmproto": CrossModalProto,
    "fedcm_moon": FedCMMoon,
}

def get_strategy(algorithm: str, **kwargs):
    try:
        strategy_cls = STRATEGY_REGISTRY[algorithm]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Available: {list(STRATEGY_REGISTRY)}"
        )
    strategy = strategy_cls(**kwargs)
    return strategy
