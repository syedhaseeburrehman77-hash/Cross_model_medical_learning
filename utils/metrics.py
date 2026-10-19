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

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return loss, accuracy, precision, recall, f1, sensitivity, specificity

def save_results(metrics, dataset, filename=None):
    import csv
    from pathlib import Path

    if filename is None:
        filename = RESULTS_FILE

    row = {
        "Dataset": dataset,
        "Algorithm": metrics.get("algorithm", ""),
        "Loss": metrics.get("eval_loss", metrics.get("loss", "")),
        "Accuracy": metrics.get("eval_acc", metrics.get("accuracy", "")),
        "Precision": metrics.get("eval_pre", metrics.get("precision", "")),
        "Recall": metrics.get("eval_rec", metrics.get("recall", "")),
        "F1-score": metrics.get("eval_f1", metrics.get("f1", "")),
        "Sensitivity": metrics.get("eval_sen", metrics.get("sensitivity", "")),
        "Specificity": metrics.get("eval_spec", metrics.get("specificity", "")),
    }

    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)
