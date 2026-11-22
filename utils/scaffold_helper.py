import tomllib
from pathlib import Path

def get_num_supernodes(default: int = 4) -> int:
    """Safely return the number of supernodes from ~/.flwr/config.toml if present, else default."""
    try:
        config_path = Path.home() / ".flwr" / "config.toml"
        if config_path.is_file():
            with config_path.open("rb") as file:
                config = tomllib.load(file)
            active_profile = config.get("superlink", {}).get("default", "default")
            profile = config.get("superlink", {}).get(active_profile, {})
            opts = profile.get("options", {})
            if "num-supernodes" in opts:
                return int(opts["num-supernodes"])
    except Exception:
        pass
    return default


def get_trainable_parameter_names(model):
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

