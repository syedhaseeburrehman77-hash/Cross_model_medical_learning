import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import csv
import os
from datetime import datetime
from pathlib import Path

def evaluate_model(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_preds)

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    unique_classes = np.unique(y_true)
    if len(unique_classes) <= 2 and set(unique_classes).issubset({0, 1}):
        # Binary classification (e.g. Brain Tumor Healthy vs Tumor)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # Multi-class classification (e.g. CIFAR-10)
        sensitivity = recall  # Macro-averaged sensitivity equals macro-recall
        num_classes = max(10, int(np.max(unique_classes)) + 1)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        specificities = []
        for i in range(num_classes):
            denom = np.sum(cm) - np.sum(cm[i, :])
            if denom > 0:
                tn_i = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                fp_i = np.sum(cm[:, i]) - cm[i, i]
                specificities.append(tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0.0)
        specificity = float(np.mean(specificities)) if specificities else 0.0

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