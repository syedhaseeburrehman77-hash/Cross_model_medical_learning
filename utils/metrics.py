import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import csv
import os
from datetime import datetime
from pathlib import Path


def _compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    """Compute aggregate accuracy, macro precision/recall/F1, sensitivity, and specificity."""
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    if num_classes <= 2:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = recall
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        specificities = []
        for i in range(num_classes):
            denom = np.sum(cm) - np.sum(cm[i, :])
            if denom > 0:
                tn_i = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                fp_i = np.sum(cm[:, i]) - cm[i, i]
                specificities.append(tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0.0)
        specificity = float(np.mean(specificities)) if specificities else 0.0

    return accuracy, precision, recall, f1, sensitivity, specificity


def _print_and_save_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_losses: np.ndarray,
    modalities: list,
    num_classes: int,
    round_num: int | None = None,
    algorithm: str | None = None,
) -> None:
    """Print and optionally persist class-wise and modality-wise metrics for paper comparison."""
    header = f"Round {round_num}" if round_num is not None else "Evaluation"
    algo_tag = f" | {algorithm}" if algorithm else ""
    print(f"\n--- [Class-Wise & Modality-Wise Breakdown | {header}{algo_tag}] ---")

    rows_to_save = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Class-wise metrics
    print("[Class-Wise]")
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int)
        y_pred_c = (y_pred == c).astype(int)
        support_c = int(y_true_c.sum())
        pre_c = float(precision_score(y_true_c, y_pred_c, zero_division=0))
        rec_c = float(recall_score(y_true_c, y_pred_c, zero_division=0))
        f1_c = float(f1_score(y_true_c, y_pred_c, zero_division=0))
        loss_c = float(np.mean(sample_losses[y_true == c])) if support_c > 0 else 0.0
        print(f"  Class {c} (n={support_c}): Loss={loss_c:.4f} | Pre={pre_c:.4f} | Rec={rec_c:.4f} | F1={f1_c:.4f}")
        rows_to_save.append({
            "Algorithm": algorithm or "",
            "Round": round_num if round_num is not None else "",
            "Timestamp": timestamp,
            "Group_Type": "class",
            "Group_Name": f"class_{c}",
            "Support": support_c,
            "Loss": loss_c,
            "Accuracy": rec_c,
            "Precision": pre_c,
            "Recall": rec_c,
            "F1_Score": f1_c,
            "Sensitivity": rec_c if c == 1 else "",
            "Specificity": rec_c if c == 0 else "",
        })

    # 2. Modality-wise metrics
    if modalities:
        mod_arr = np.asarray(modalities)
        print("[Modality-Wise]")
        for m in sorted(set(modalities)):
            mask = mod_arr == m
            n_m = int(mask.sum())
            y_t_m = y_true[mask]
            y_p_m = y_pred[mask]
            loss_m = float(np.mean(sample_losses[mask])) if n_m > 0 else 0.0
            acc_m, pre_m, rec_m, f1_m, sen_m, spec_m = _compute_classification_metrics(y_t_m, y_p_m, num_classes)
            if num_classes <= 2:
                print(
                    f"  {m:<12} (n={n_m}): Loss={loss_m:.4f} | Acc={acc_m:.4f} | "
                    f"F1={f1_m:.4f} | Sen={sen_m:.4f} | Spec={spec_m:.4f}"
                )
            else:
                print(
                    f"  {m:<12} (n={n_m}): Loss={loss_m:.4f} | Acc={acc_m:.4f} | "
                    f"F1={f1_m:.4f} | Pre={pre_m:.4f} | Rec={rec_m:.4f}"
                )
            rows_to_save.append({
                "Algorithm": algorithm or "",
                "Round": round_num if round_num is not None else "",
                "Timestamp": timestamp,
                "Group_Type": "modality",
                "Group_Name": m,
                "Support": n_m,
                "Loss": loss_m,
                "Accuracy": acc_m,
                "Precision": pre_m,
                "Recall": rec_m,
                "F1_Score": f1_m,
                "Sensitivity": sen_m,
                "Specificity": spec_m,
            })

    if algorithm and rows_to_save:
        detail_file = Path(f"results/detailed_metrics_{algorithm}.csv")
        detail_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = detail_file.is_file()
        with open(detail_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_to_save[0].keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows_to_save)


def evaluate_model(
    net,
    testloader,
    device,
    print_breakdown: bool = False,
    round_num: int | None = None,
    algorithm: str | None = None,
):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    all_labels = []
    all_preds = []
    all_losses = []
    all_modalities = []
    num_classes = 2
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            num_classes = outputs.shape[1]
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

            per_sample_loss = F.cross_entropy(outputs, labels, reduction="none")
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_losses.extend(per_sample_loss.cpu().numpy())
            if "modality" in batch:
                all_modalities.extend(list(batch["modality"]))

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_preds)
    sample_losses = np.asarray(all_losses)

    _, precision, recall, f1, sensitivity, specificity = _compute_classification_metrics(
        y_true, y_pred, num_classes
    )

    if print_breakdown:
        _print_and_save_breakdown(
            y_true=y_true,
            y_pred=y_pred,
            sample_losses=sample_losses,
            modalities=all_modalities,
            num_classes=num_classes,
            round_num=round_num,
            algorithm=algorithm,
        )

    return loss, accuracy, precision, recall, f1, sensitivity, specificity

def save_metrics(
    loss,
    accuracy,
    precision,
    recall,
    f1,
    sensitivity,
    specificity,
    algorithm,
    filename=None,
):
    """Save evaluation metrics to a CSV file."""

    if filename is None:
        filename = f"results/evaluation_metrics_{algorithm}.csv"

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.isfile(filename)

    metrics = {
        "Algorithm": algorithm,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Loss": loss,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }

    with open(filename, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics)

    print(f"Metrics saved to {filename}")