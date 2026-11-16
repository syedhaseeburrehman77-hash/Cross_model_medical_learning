"""pytorchexample: A Flower / PyTorch app."""
import torch
import json
import numpy as np
from pathlib import Path
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from task import load_data
from utils.metrics import evaluate_model

from algorithms import get_trainer

from models import create_model

from utils.fedbn_helper import get_fedbn_state_dict, get_bn_state_dict, load_bn_state_dict
from utils.fedpara_helper import convert_to_fedpara
from utils.fedrep_helper import get_fedrep_head_state_dict, get_fedrep_shared_state_dict, load_fedrep_head_state_dict
from utils.moon_helper import prepare_moon_model
from utils.fedper_helper import get_fedper_head_state_dict, get_fedper_shared_state_dict, load_fedper_head_state_dict
from utils.floco_helper import convert_to_floco
from utils.prototype_helper import compute_class_prototypes
from utils.fedpac_helper import load_classifier_head
from utils.fedlc_helper import get_fedlc_head_state_dict, get_fedlc_shared_state_dict, load_fedlc_head_state_dict

def check_finite_state(state, stage):
    for name, tensor in state.items():
        if torch.is_tensor(tensor) and (
            torch.is_floating_point(tensor)
            or torch.is_complex(tensor)
        ):
            if not torch.isfinite(tensor).all():
                raise RuntimeError(
                    f"[{stage}] NaN/Inf detected in: {name}"
                )

# Flower ClientApp
app = ClientApp()
@app.train()

def train(msg: Message, context: Context):
    """Train the model on local data."""

    algorithm = context.run_config["algorithm"]
    model_name = context.run_config["model_name"]
    scenario = context.run_config.get("scenario", "binary")
    num_classes = int(context.run_config.get("num-classes", 2))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_model = None
    
    # Create model
    model = create_model(model_name, num_classes=num_classes) 
    
    if algorithm == "fedpara":
        convert_to_fedpara(model)

    elif algorithm == "moon":
        prepare_moon_model(
            model,
            projection_dim=context.run_config.get("moon-projection-dim", 256),
        )

    elif algorithm == "floco":
        endpoints = context.run_config.get("endpoints", 3)
        seed = context.run_config.get("seed", 0)
        model = convert_to_floco(model, endpoints=endpoints, seed=seed)

    # Load received parameters
    received_state= msg.content["arrays"].to_torch_state_dict()
    check_finite_state(received_state, "received global model")

    scaffold_parameter_names = []
    scaffold_server_cv = None
    scaffold_client_cv = None

    if algorithm == "scaffold":
        server_cv_prefix = "__scaffold_server_cv__"

        scaffold_parameter_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]

        # Extract server control variates from the received arrays.
        scaffold_server_cv = []

        for name in scaffold_parameter_names:
            cv_key = server_cv_prefix + name

            if cv_key not in received_state:
                raise ValueError(
                    f"Missing server control variate: {cv_key}"
                )

            scaffold_server_cv.append(
                received_state.pop(cv_key)
            )

        # Restore this client's control variates from its persistent state.
        if "scaffold_client_cv" in context.state:
            saved_cv = (
                context.state["scaffold_client_cv"]
                .to_torch_state_dict()
            )

            scaffold_client_cv = [
                saved_cv[name]
                for name in scaffold_parameter_names
            ]

        else:
            # First participation: initialize client control variates to zero.
            named_parameters = dict(model.named_parameters())

            scaffold_client_cv = [
                torch.zeros_like(named_parameters[name].detach())
                for name in scaffold_parameter_names
            ]

    # FedMLB: create a teacher from the received global parameters.
    elif algorithm == "fedmlb":
        global_model = create_model(model_name, num_classes=num_classes)
        global_model.load_state_dict(received_state)
        global_model.to(device)
        global_model.eval()
        for parameter in global_model.parameters():
            parameter.requires_grad_(False)

    alpha_state = None
    if algorithm == "fedmeta_meta_sgd":
        alpha_state = {k[6:]: v for k, v in received_state.items() if k.startswith("alpha.")}
        model_state = {k: v for k, v in received_state.items() if not k.startswith("alpha.")}
        model.load_state_dict(model_state)

    elif algorithm == "fedbn":
        # Load shared/global parameters
        model.load_state_dict(received_state, strict=False,)

        # Restore this client's local BatchNorm state
        if "fedbn_bn_state" in context.state:
            print("Restoring local BN state")
            bn_state = context.state["fedbn_bn_state"].to_torch_state_dict()
            load_bn_state_dict(model, bn_state)
        else:
            print("No previous BN state")

    elif algorithm == "fedrep":
        model.load_state_dict(received_state, strict=False)

        if "fedrep_head_state" in context.state:
            head_state = context.state["fedrep_head_state"].to_torch_state_dict()
            load_fedrep_head_state_dict(model, head_state)
            print(f"[FedRep] Client {context.node_config['partition-id']}: local head RESTORED")
            
        else:
           print(f"[FedRep] Client {context.node_config['partition-id']}: no previous head; using initialized head")  

    
    elif algorithm == "fedper":
        model.load_state_dict(received_state, strict=False)

        if "fedper_head_state" in context.state:
            head_state = context.state["fedper_head_state"].to_torch_state_dict()
            load_fedper_head_state_dict(model, head_state)
            print(f"[FedPer] Client {context.node_config['partition-id']}: local head RESTORED")
        else:
            print(f"[FedPer] Client {context.node_config['partition-id']}: no previous head; using initialized head")

    elif algorithm == "fedpac":
        model.load_state_dict(received_state, strict=False)
        collab_file = Path("results/fedpac") / f"client_{context.node_config['partition-id']}_collab_head.pt"
        if collab_file.exists():
            try:
                head_state = torch.load(collab_file, weights_only=False)
                load_classifier_head(model, head_state)
                print(f"[FedPAC] Client {context.node_config['partition-id']}: collaborative head LOADED from server")
            except Exception:
                pass
        elif "fedpac_head_state" in context.state:
            head_state = context.state["fedpac_head_state"].to_torch_state_dict()
            load_classifier_head(model, head_state)

    elif algorithm == "fedlc":
        model.load_state_dict(received_state, strict=False)
        if "fedlc_head_state" in context.state:
            head_state = context.state["fedlc_head_state"].to_torch_state_dict()
            load_fedlc_head_state_dict(model, head_state)
            print(f"[FedLC] Client {context.node_config['partition-id']}: local head & calibration RESTORED")
        else:
            print(f"[FedLC] Client {context.node_config['partition-id']}: initial head & calibration initialized")

    else:
        model.load_state_dict(received_state)
        

    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    if num_partitions == 2:
        modality = "MRI" if partition_id == 0 else "CT"
    elif num_partitions == 4:
        modality = "MRI" if partition_id in (0, 1) else "CT"
    else:
        modality = f"Client_{partition_id}"

    batch_size = context.run_config["batch-size"]
    image_size = int(context.run_config.get("image-size", 224))
    dataset_name = context.run_config.get("dataset-name", None)
    dataset_path = context.run_config.get("dataset-path", None)
    trainloader, testloader = load_data(partition_id, num_partitions, batch_size, image_size=image_size, dataset_name=dataset_name, dataset_path=dataset_path, scenario=scenario,)

    algorithm = context.run_config["algorithm"]
    cfg = msg.content.get("config", {})

    if algorithm == "scaffold":
        trainer = get_trainer(
            algorithm,
            server_cv=scaffold_server_cv,
            client_cv=scaffold_client_cv,
        )
    else:
        # Load global prototypes for FedPAC feature alignment
        fedpac_global_protos = None
        if algorithm == "fedpac":
            proto_files = sorted(list(Path("results/fedpac").glob("global_prototypes_round_*.pt"))) if Path("results/fedpac").exists() else []
            if proto_files:
                try:
                    p_data = torch.load(proto_files[-1], weights_only=False)
                    fedpac_global_protos = p_data.get("global_prototypes")
                    print(f"[FedPAC] Client {partition_id}: loaded {len(fedpac_global_protos)} global prototypes for feature alignment")
                except Exception:
                    pass

        head_name = "classifier" if hasattr(model, "classifier") else "fc3"
        trainer = get_trainer(
        algorithm,
        head_module_name=head_name,
        proximal_mu=msg.content["config"].get("proximal_mu", 0.0),
        head_epochs=msg.content["config"].get("fedlc-head-epochs", msg.content["config"].get("fedrep-head-epochs", 1)),
        mu=context.run_config.get("moon-mu", 1.0),
        temperature=context.run_config.get("moon-temperature", 0.5),
        momentum=context.run_config.get("moon-momentum", 0.9),
        weight_decay=context.run_config.get("moon-weight-decay", 1e-5),
        global_model=global_model,
        lambda_1=context.run_config.get("fedmlb-lambda-1", 1.0),
        lambda_2=context.run_config.get("fedmlb-lambda-2", 1.0),
        alpha_state=alpha_state,
        alpha=context.run_config.get("fedlc-alpha", context.run_config.get("fedmeta-alpha", 0.01)),
        gradient_step=context.run_config.get("fedmeta-gradient-step", 5),
        support_ratio=context.run_config.get("fedmeta-support-ratio", 0.2),
        pers_lamda=context.run_config.get("pers_lamda", 0.0),
        proto_lambda=context.run_config.get("proto-lambda", 1.0),
        fedpac_lambda=context.run_config.get("fedpac-lambda", 1.0),
        global_prototypes=fedpac_global_protos if algorithm == "fedpac" else None,
        pcs=context.run_config.get("fedlc-pcs", True),
        hc=context.run_config.get("fedlc-hc", True),
    )

    # Call the training function
    if algorithm == "moon":
        previous_state_dict = None
        if "moon_previous_state" in context.state:
            previous_state_dict = context.state["moon_previous_state"].to_torch_state_dict()

        train_loss = trainer.train(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
            previous_state_dict=previous_state_dict,
        )
        # Save trained local state for next round's contrastive loss
        context.state["moon_previous_state"] = ArrayRecord(
                {k: v.detach().cpu() for k, v in model.state_dict().items()})

    elif algorithm == "floco":
        # Extract subregion parameters if provided by server (from round tau onwards)
        subregion_params = None
        if "center" in cfg and "radius" in cfg:
            center_val = np.array(json.loads(cfg["center"]))
            radius_val = float(cfg["radius"])
            subregion_params = (center_val, radius_val)
        
        train_loss = trainer.train(
            model,
            trainloader,
            context.run_config["local-epochs"],
            cfg.get("lr", 0.01),
            device,
            subregion_parameters=subregion_params,
        )

    elif algorithm == "scaffold":
        print(
        f"[SCAFFOLD DEBUG] algorithm={algorithm}, "
        f"server_cv_is_none={trainer.server_cv is None}, "
        f"client_cv_is_none={trainer.client_cv is None}, "
        f"server_cv_count={len(trainer.server_cv) if trainer.server_cv is not None else 'None'}, "
        f"client_cv_count={len(trainer.client_cv) if trainer.client_cv is not None else 'None'}"
    )
        train_loss = trainer.train(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )

        # Retrieve model and control-variate changes.
        model_delta, client_cv_delta = trainer.get_updates()
        client_cv_new = trainer.get_new_client_cv()

        # Persist updated client control variates between rounds.
        context.state["scaffold_client_cv"] = ArrayRecord({
            name: value
            for name, value in zip(
                scaffold_parameter_names, client_cv_new
            )
        })

    elif algorithm in ("fedavg_meta", "fedmeta_maml", "fedmeta_meta_sgd"):
        train_loss = trainer.train(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )

    else:
        train_loss = trainer.train(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
            valloader=testloader,
            patience=5,
        )

        check_finite_state(
            model.state_dict(),
            f"client {partition_id} after local training"
        )

    # Clean up teacher model to conserve VRAM on laptop GPUs
    if algorithm == "fedmlb":
        del global_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if algorithm == "fedbn":
        bn_state = get_bn_state_dict(model)
        context.state["fedbn_bn_state"] = ArrayRecord(bn_state)
        print("Saved local BN state")

    elif algorithm == "fedrep":
        head_state = get_fedrep_head_state_dict(model)
        context.state["fedrep_head_state"] = ArrayRecord(head_state)
        print(f"Client {partition_id}: saved personalized head "
            f"with {len(head_state)} tensors")

    elif algorithm == "fedper":
        head_state = get_fedper_head_state_dict(model)
        context.state["fedper_head_state"] = ArrayRecord(head_state)
        print(f"[FedPer] Client {partition_id}: saved personalized head "
            f"with {len(head_state)} tensors")

    elif algorithm == "fedlc":
        head_state = get_fedlc_head_state_dict(model)
        context.state["fedlc_head_state"] = ArrayRecord(head_state)
        print(f"[FedLC] Client {partition_id}: saved personalized head/calibration "
            f"with {len(head_state)} tensors")

    # Class Prototypes only needed for prototype-based algorithms unless explicitly enabled
    compute_prototypes = context.run_config.get("compute-prototypes", False) or algorithm in ("fedproto", "fedpac")
    if compute_prototypes:
        try:
            client_protos, class_counts = compute_class_prototypes(model, trainloader, device)
            proto_dir = Path("results/prototypes")
            proto_dir.mkdir(parents=True, exist_ok=True)
            
            # Save client prototypes
            proto_data = {
                "prototypes": client_protos,
                "counts": class_counts,
                "partition_id": partition_id,
                "modality": modality,
            }
            torch.save(proto_data, proto_dir / f"client_{partition_id}_latest_prototype.pt")
            
            # Log clear summary for the client
            print(f"[Prototype] Client {partition_id} ({modality}): computed prototypes for classes {sorted(list(client_protos.keys()))}")
            for c in sorted(list(client_protos.keys())):
                p = client_protos[c]
                c_name = "Tumor" if c == 1 else ("Healthy" if c == 0 else f"Class_{c}")
                print(f"  └── Class {c} ({c_name}): count={class_counts.get(c, 0)}, dim={p.shape[0]}, norm={p.norm().item():.4f}")
        except Exception as e:
            print(f"[Prototype Warning] Failed to compute client {partition_id} prototypes: {e}")

    # Construct and return reply Message
    if algorithm == "fedbn":
        model_record = ArrayRecord(get_fedbn_state_dict(model))
    elif algorithm == "fedrep":
        model_record = ArrayRecord(get_fedrep_shared_state_dict(model))
    elif algorithm == "fedper":
        model_record = ArrayRecord(get_fedper_shared_state_dict(model))
    elif algorithm in ("fedmeta_maml", "fedmeta_meta_sgd"):
        model_record = ArrayRecord(trainer.get_gradients())    
    elif algorithm == "scaffold":
        updates = {}
        # Send model parameter changes (y_i - x)
        for name, value in zip(scaffold_parameter_names, model_delta):
            updates["__scaffold_model_delta__" + name] = value

        # Send client control-variate changes (c_i_new - c_i)
        for name, value in zip(scaffold_parameter_names, client_cv_delta):
            updates["__scaffold_client_cv_delta__" + name] = value
        model_record = ArrayRecord(updates)

    else:
        model_record = ArrayRecord(model.state_dict())


    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "partition-id": partition_id,
    }
    
    if hasattr(trainer, "get_metrics"):
        metrics.update(trainer.get_metrics())

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    
    algorithm = context.run_config["algorithm"]
    model_name = context.run_config["model_name"]
    scenario = context.run_config.get("scenario", "binary")
    num_classes = int(context.run_config.get("num-classes", 2))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model(model_name, num_classes=num_classes)
    if algorithm == "fedpara":
        convert_to_fedpara(model)

    elif algorithm == "moon":
        prepare_moon_model(
            model,
            projection_dim=context.run_config.get("moon-projection-dim", 256),
        )
    
    elif algorithm == "floco":
        endpoints = context.run_config.get("endpoints", 3)
        seed = context.run_config.get("seed", 0)
        model = convert_to_floco(model, endpoints=endpoints, seed=seed)
   
    received_state = msg.content["arrays"].to_torch_state_dict()
    check_finite_state(
    received_state,
    f"client {context.node_config['partition-id']} received model for evaluation"
    )

    if algorithm == "fedmeta_meta_sgd":
        model_state = {k: v for k, v in received_state.items() if not k.startswith("alpha.")}
        model.load_state_dict(model_state)

    elif algorithm == "fedbn":
        model.load_state_dict(received_state, strict=False,)
        if "fedbn_bn_state" in context.state:
            bn_state = context.state["fedbn_bn_state"].to_torch_state_dict()
            load_bn_state_dict(model, bn_state)

    elif algorithm == "fedrep":
        model.load_state_dict(received_state, strict=False)
        if "fedrep_head_state" in context.state:
            head_state = context.state["fedrep_head_state"].to_torch_state_dict()
            load_fedrep_head_state_dict(model, head_state)

    elif algorithm == "fedper":
        model.load_state_dict(received_state, strict=False)
        if "fedper_head_state" in context.state:
            head_state = context.state["fedper_head_state"].to_torch_state_dict()
            load_fedper_head_state_dict(model, head_state)

    elif algorithm == "fedpac":
        model.load_state_dict(received_state, strict=False)
        collab_file = Path("results/fedpac") / f"client_{context.node_config['partition-id']}_collab_head.pt"
        if collab_file.exists():
            try:
                head_state = torch.load(collab_file, weights_only=False)
                load_classifier_head(model, head_state)
            except Exception:
                pass
        elif "fedpac_head_state" in context.state:
            head_state = context.state["fedpac_head_state"].to_torch_state_dict()
            load_classifier_head(model, head_state)

    elif algorithm == "fedlc":
        model.load_state_dict(received_state, strict=False)
        if "fedlc_head_state" in context.state:
            head_state = context.state["fedlc_head_state"].to_torch_state_dict()
            load_fedlc_head_state_dict(model, head_state)

    else:
        model.load_state_dict(received_state,)

    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    if num_partitions == 2:
        modality = "MRI" if partition_id == 0 else "CT"
    elif num_partitions == 4:
        modality = "MRI" if partition_id in (0, 1) else "CT"
    else:
        modality = f"Client_{partition_id}"

    batch_size = context.run_config["batch-size"]
    image_size = int(context.run_config.get("image-size", 224))
    dataset_name = context.run_config.get("dataset-name", None)
    dataset_path = context.run_config.get("dataset-path", None)
    _, valloader = load_data(partition_id, num_partitions, batch_size, image_size=image_size, dataset_name=dataset_name, dataset_path=dataset_path, scenario=scenario)

    cfg = msg.content.get("config", {})

    if algorithm in ("fedavg_meta", "fedmeta_maml", "fedmeta_meta_sgd"):
        trainer = get_trainer(
            algorithm,
            alpha=context.run_config.get("fedmeta-alpha"),
            gradient_step=context.run_config.get("fedmeta-gradient-step"),
            support_ratio=context.run_config.get("fedmeta-support-ratio"),
        )
        eval_loss, eval_acc, eval_precision, eval_recall, eval_f1, eval_sensitivity, eval_specificity = trainer.evaluate(model, valloader, device)

    elif algorithm == "floco":
        subregion_params = None
        cfg = msg.content.get("config", {})
        if "center" in cfg and "radius" in cfg:
            center_val = np.array(json.loads(cfg["center"]))
            radius_val = float(cfg["radius"])
            subregion_params = (center_val, radius_val)

        if hasattr(model, "subregion_parameters"):
            model.subregion_parameters = subregion_params

        eval_loss, eval_acc, eval_precision, eval_recall, eval_f1, eval_sensitivity, eval_specificity = evaluate_model(
            model,
            valloader,
            device,
        )

    else:
        # Call the evaluation function
        eval_loss, eval_acc, eval_precision, eval_recall, eval_f1, eval_sensitivity, eval_specificity = evaluate_model(
            model,
            valloader,
            device,
        )

                # Check global model parameters and buffers
        for name, tensor in model.state_dict().items():
            if torch.is_floating_point(tensor) or torch.is_complex(tensor):
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(
                        f"Global model contains NaN/Inf before evaluation: {name}"
                    )

        # Inspect the first evaluation batch
        batch = next(iter(valloader))
        images = batch["img"].to(device)

        if not torch.isfinite(images).all():
            raise RuntimeError("Global evaluation images contain NaN/Inf")

        print("Global model state is finite")
        print("Input images are finite")
        print("Input shape:", tuple(images.shape))
        print("Input range:", images.min().item(), images.max().item())

        model.eval()
        with torch.no_grad():
            outputs = model(images)

        print("Output shape:", tuple(outputs.shape))
        print("Output finite:", torch.isfinite(outputs).all().item())

        if torch.isfinite(outputs).any():
            finite_outputs = outputs[torch.isfinite(outputs)]
            print(
                "Finite output range:",
                finite_outputs.min().item(),
                finite_outputs.max().item()
            )
        
    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "eval_pre": eval_precision,
        "eval_rec": eval_recall,
        "eval_f1": eval_f1,
        "eval_sen": eval_sensitivity,
        "eval_spec": eval_specificity,
        "num-examples": len(valloader.dataset),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)