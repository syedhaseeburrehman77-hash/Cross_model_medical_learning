import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from task import load_centralized_dataset, get_dataset
from utils.metrics import evaluate_model, save_metrics
from utils.data_distribution import export_data_distribution

from algorithms.server_strategies import get_strategy

from models import create_model

from utils.fedbn_helper import get_fedbn_state_dict
from utils.fedpara_helper import convert_to_fedpara
from utils.fedrep_helper import get_fedrep_shared_state_dict
from utils.moon_helper import prepare_moon_model
from utils.fedper_helper import get_fedper_shared_state_dict
from utils.fedlc_helper import get_fedlc_shared_state_dict
from utils.floco_helper import convert_to_floco
from utils.scaffold_helper import get_num_supernodes, get_trainable_parameter_names
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords
from utils.prototype_helper import aggregate_and_export_round_prototypes
import wandb
from pathlib import Path

# Create ServerApp
app = ServerApp()
model_name = None
algorithm = None
moon_projection_dim = None
floco_endpoints = None
image_size = 224
@app.main()

def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    global model_name, algorithm, moon_projection_dim, floco_endpoints, image_size
    algorithm = context.run_config["algorithm"]
    # Clear the evaluation metrics CSV once at the beginning of this run
    metrics_file = Path(f"results/evaluation_metrics_{algorithm}.csv")
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    if metrics_file.exists():
        metrics_file.unlink()

    print(f"Started fresh metrics CSV: {metrics_file}")


    moon_projection_dim = context.run_config.get("moon-projection-dim", 84)
    floco_endpoints = context.run_config.get("endpoints", 3)
    image_size = int(context.run_config.get("image-size", 224))

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Model selected from config
    model_name = context.run_config["model_name"]

    run = wandb.init(
        project="federated-learning",
        name=f"{algorithm}-{model_name}",
        config={
            "algorithm": algorithm,
            "model_name": model_name,
            "num_rounds": num_rounds,
            "learning_rate": lr,
            "fraction_evaluate": fraction_evaluate,
            "num_clients": get_num_supernodes(),
            "batch_size": context.run_config["batch-size"],
            "local_epochs": context.run_config["local-epochs"],
        },
    )

    wandb.define_metric("round")
    wandb.define_metric("*", step_metric="round")

    # Load global model
    global_model = create_model(model_name)

    if algorithm == "fedpara":
        convert_to_fedpara(global_model) 

    elif algorithm == "moon":
        prepare_moon_model(global_model, projection_dim=moon_projection_dim)

    elif algorithm == "floco":
        seed = context.run_config.get("seed", 0)
        global_model = convert_to_floco(global_model, endpoints=floco_endpoints, seed=seed)   

    if algorithm == "fedbn":
        arrays = ArrayRecord(get_fedbn_state_dict(global_model))

    elif algorithm == "fedrep":
        arrays = ArrayRecord(get_fedrep_shared_state_dict(global_model))

    elif algorithm == "fedper":
        arrays = ArrayRecord(get_fedper_shared_state_dict(global_model))

    elif algorithm == "fedlc":
        arrays = ArrayRecord(get_fedlc_shared_state_dict(global_model))
    
    else:
        arrays = ArrayRecord(global_model.state_dict())

    # Track rounds for client-side training and evaluation
    train_round = 0
    def wandb_train_metrics_aggr(records, weighting_key):
        nonlocal train_round
        train_round += 1
        # Flower weighted aggregation of client training metrics
        metrics = aggregate_metricrecords(records, weighting_key)

        log_data = {"round": train_round}
        for k, v in metrics.items():
            # Exclude raw partition-id from graphs
            if k != "partition-id" and isinstance(v, (int, float)):
                log_data[f"train/{k}"] = float(v)
        wandb.log(log_data)
        return metrics

    eval_round = 0
    def wandb_eval_metrics_aggr(records, weighting_key):
        nonlocal eval_round
        eval_round += 1
        # Flower weighted aggregation of client validation metrics
        metrics = aggregate_metricrecords(records, weighting_key)

        wandb.log({
            "round": eval_round,
            "fed_eval/accuracy": float(metrics.get("eval_acc", 0.0)),
            "fed_eval/loss": float(metrics.get("eval_loss", 0.0)),
            "fed_eval/precision": float(metrics.get("eval_pre", 0.0)),
            "fed_eval/recall": float(metrics.get("eval_rec", 0.0)),
            "fed_eval/f1": float(metrics.get("eval_f1", 0.0)),
            "fed_eval/sensitivity": float(metrics.get("eval_sen", 0.0)),
            "fed_eval/specificity": float(metrics.get("eval_spec", 0.0)),
        })
        return metrics

    # Build kwargs relevant to whichever strategy is picked;
    # extras are ignored by strategies that don't use them
    strategy_kwargs = {
        "fraction_evaluate": context.run_config["fraction-evaluate"],
        "min_available_nodes": context.run_config["min-available-clients"],
        "train_metrics_aggr_fn": wandb_train_metrics_aggr,
        "evaluate_metrics_aggr_fn": wandb_eval_metrics_aggr,
    }

    if algorithm == "fedprox":
        strategy_kwargs["proximal_mu"] = context.run_config["proximal-mu"]

    elif algorithm == "fedavgm":
        strategy_kwargs["server_learning_rate"] = context.run_config["server-learning-rate"]
        strategy_kwargs["server_momentum"] = context.run_config["server-momentum"]

    elif algorithm in ("fedadam", "fedyogi"):
        strategy_kwargs["eta"] = context.run_config["server-learning-rate"]
        strategy_kwargs["eta_l"] = context.run_config["learning-rate"]
        strategy_kwargs["beta_1"] = context.run_config["beta-1"]
        strategy_kwargs["beta_2"] = context.run_config["beta-2"]
        strategy_kwargs["tau"] = context.run_config["tau"]

    elif algorithm == "fedadagrad":
        strategy_kwargs["eta"] = context.run_config["server-learning-rate"]
        strategy_kwargs["eta_l"] = context.run_config["learning-rate"]
        strategy_kwargs["tau"] = context.run_config["tau"]

    elif algorithm == "fednova":
        strategy_kwargs["server_learning_rate"] = context.run_config.get("server-learning-rate")
        strategy_kwargs["server_momentum"] = context.run_config.get("server-momentum")
        
    elif algorithm in ("fedmeta_maml", "fedmeta_meta_sgd"):
        strategy_kwargs["algo"] = algorithm
        strategy_kwargs["model_name"] = model_name
        strategy_kwargs["beta"] = context.run_config.get("fedmeta-beta")
        strategy_kwargs["alpha_init"] = context.run_config.get("fedmeta-alpha")

    elif algorithm == "floco":
        strategy_kwargs["endpoints"] = context.run_config.get("endpoints")
        strategy_kwargs["tau"] = context.run_config.get("tau")
        strategy_kwargs["rho"] = context.run_config.get("rho")
        strategy_kwargs["fraction_fit"] = context.run_config.get("fraction-fit")

    elif algorithm == "scaffold":
        strategy_kwargs["parameter_names"] =  get_trainable_parameter_names(global_model)
        strategy_kwargs["num_clients"] = get_num_supernodes()

    elif algorithm == "fedpac":
        strategy_kwargs["fedpac_lambda"] = context.run_config.get("fedpac-lambda")
        strategy_kwargs["fedpac_tau"] = context.run_config.get("fedpac-tau")

    elif algorithm == "fedlc":
        strategy_kwargs["alpha"] = context.run_config.get("fedlc-alpha")
        strategy_kwargs["pcs"] = context.run_config.get("fedlc-pcs")
        strategy_kwargs["hc"] = context.run_config.get("fedlc-hc")

    strategy = get_strategy(algorithm, **strategy_kwargs)

    if algorithm in ("fedbn", "fedrep", "fedper", "fedlc"):
        evaluate_fn = None
    else:
        evaluate_fn = global_evaluate

    # Export distribution before training
    num_partitions = get_num_supernodes()
    dataset = get_dataset()

    export_data_distribution(
        dataset=dataset,
        num_partitions=num_partitions,
    )

    try:
        
        # Start strategy, run for `num_rounds`
        result = strategy.start(
                grid=grid,
                initial_arrays=arrays,
                train_config=ConfigRecord({"lr": lr}),
                num_rounds=num_rounds,
                evaluate_fn=evaluate_fn,
            )

        if context.run_config["save-model"]:
                print("\nSaving final model to disk...")
                state_dict = result.arrays.to_torch_state_dict()

                if algorithm in ("fedper", "fedrep", "fedpac", "fedlc"):
                    torch.save(state_dict, f"{algorithm}_shared_state.pt")
                    print(
                        f"Saved {algorithm} shared parameters only. "
                        "Client-specific heads are not included."
                    )
                else:
                    torch.save(state_dict, "final_model.pt")
    
    finally:
        wandb.finish()

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
   
    model = create_model(model_name)  

    if algorithm == "fedpara":
        convert_to_fedpara(model)

    elif algorithm == "moon":
        prepare_moon_model(model, projection_dim=moon_projection_dim)

    elif algorithm == "floco":
        seed = 0
        endpoints = floco_endpoints or 3
        model = convert_to_floco(model, endpoints=floco_endpoints, seed=seed)

    # Use strict=True for models that don't have partial loading (e.g. resnet18_mlb, fedmlb).
    # Use strict=False only for algorithms that send partial state dicts (fedpara, floco, moon).
    needs_strict_false = algorithm in ("fedpara", "floco", "moon", "fedbn", "fedrep", "fedper", "fedpac", "fedlc")
    model.load_state_dict(arrays.to_torch_state_dict(), strict=not needs_strict_false)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset(image_size=image_size)

    # Evaluate the global model on the test set
    test_loss, test_acc, test_precision, test_recall, test_f1, test_sensitivity, test_specificity = evaluate_model(model, test_dataloader, device)

    # Return the evaluation metrics
    metrics= {"accuracy": test_acc,
                        "loss": test_loss,
                        "test_pre": test_precision,
                        "test_rec": test_recall,
                        "test_f1": test_f1,
                        "test_sen": test_sensitivity,
                        "test_spec": test_specificity,}

     # Log global evaluation metrics to W&B
    wandb.log({
        "round": server_round,
        "global/accuracy": float(test_acc),
        "global/loss": float(test_loss),
        "global/precision": float(test_precision),
        "global/recall": float(test_recall),
        "global/f1": float(test_f1),
        "global/sensitivity": float(test_sensitivity),
        "global/specificity": float(test_specificity),
    })

    # Save the metrics
    save_metrics(
       test_loss, test_acc, test_precision, test_recall, test_f1, test_sensitivity, test_specificity, algorithm=algorithm,
    )

    # Class Prototypes Aggregation & Similarity Analysis
    try:
        num_nodes = get_num_supernodes()
        global_protos, sims = aggregate_and_export_round_prototypes(
            round_num=server_round,
            num_partitions=num_nodes,
            output_dir="results",
        )
        if global_protos:
            print(f"\n[Global Prototypes] Round {server_round}: Aggregated across {num_nodes} clients for classes {sorted(list(global_protos.keys()))}")
            for k, v in sims.items():
                print(f"  ├── Similarity {k}: {v:.4f}")
    except Exception as e:
        print(f"[Prototype Warning] Could not aggregate prototypes for round {server_round}: {e}")

    return MetricRecord(metrics)


