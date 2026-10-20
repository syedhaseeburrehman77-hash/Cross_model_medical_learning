import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import csv
import os
from datetime import datetime

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

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return loss, accuracy, precision, recall, f1, sensitivity, specificity


results_file = (f"evaluation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

def save_metrics(
    loss,
    accuracy,
    precision,
    recall,
    f1,
    sensitivity,
    specificity,
    filename=results_file
):
    """Save evaluation metrics to a CSV file."""

    file_exists = os.path.isfile(filename)

    metrics = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Loss": loss,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }

    with open(filename, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())

        # Write column names only when creating a new file
        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics)

    print(f"Metrics saved to {filename}")
