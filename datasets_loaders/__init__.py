from .registry import DATASET_REGISTRY


def create_dataset(name: str, root_dir: str, scenario: str = None):

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")

    # If milk10k, pass the chosen scenario
    if name == "milk10k" and scenario is not None:
        return DATASET_REGISTRY[name](root_dir, scenario=scenario)

    return DATASET_REGISTRY[name](root_dir)