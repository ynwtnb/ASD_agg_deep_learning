"""
Evaluation utilities for the TCN aggression prediction model.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def find_optimal_threshold(y_true, y_probs):
    """
    Finds the decision threshold that maximises F1 on a given set of labels and probabilities.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
    y_probs : np.ndarray, shape (N,)

    Returns
    -------
    threshold : float
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


def evaluate(model, loader, device, threshold=None):
    """
    Computes classification metrics on a DataLoader.

    AUPRC is the primary metric for this task due to heavy class imbalance.
    AUROC is retained for comparison with the published baseline (Goodwin et al. 2023).
    Sensitivity and specificity are the primary clinical metrics at the
    decision threshold. If threshold is None, it is selected by maximising F1
    on the predictions from this loader.

    Parameters
    ----------
    model : torch.nn.Module
    loader : torch.utils.data.DataLoader
    device : torch.device
    threshold : float or None
        Decision threshold for positive class. If None, threshold is chosen
        by maximising F1 on this loader's predictions.

    Returns
    -------
    metrics : dict
        Keys: auprc, auc_roc, accuracy, f1_binary, f1_macro,
        sensitivity, specificity, confusion_matrix, threshold.
    """
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            logits = model(signals).squeeze(1)      # (batch,)
            probs = logits.sigmoid()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    if threshold is None:
        threshold = find_optimal_threshold(all_labels, all_probs)

    all_preds = (all_probs >= threshold).astype(int)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_roc = float('nan')

    try:
        auprc = average_precision_score(all_labels, all_probs)
    except ValueError:
        auprc = float('nan')

    return {
        'auprc': auprc,
        'auc_roc': auc_roc,
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_binary': f1_score(all_labels, all_preds, average='binary', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'threshold': threshold,
    }


def evaluate_val_auprc(model, loader, device):
    """
    Computes only AUPRC on a validation loader for use as the early stopping signal.

    Parameters
    ----------
    model : torch.nn.Module
    loader : torch.utils.data.DataLoader
    device : torch.device

    Returns
    -------
    auprc : float
    """
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            probs = model(signals).squeeze(1).sigmoid()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    try:
        return float(average_precision_score(all_labels, all_probs))
    except ValueError:
        return float('nan')


def summarize_metrics(all_metrics_dict):
    """
    Computes mean and std of scalar metrics across folds.

    Parameters
    ----------
    all_metrics_dict : dict
        Keys are fold identifiers; values are metric dicts from evaluate().

    Returns
    -------
    summary : dict
        Keys: '<metric>_mean', '<metric>_std' for each scalar metric.
    """
    scalar_keys = [
        'auprc', 'auc_roc', 'accuracy',
        'f1_binary', 'f1_macro', 'sensitivity', 'specificity',
    ]
    summary = {}
    for key in scalar_keys:
        values = [
            v[key] for v in all_metrics_dict.values()
            if not np.isnan(v[key])
        ]
        if values:
            summary[f'{key}_mean'] = float(np.mean(values))
            summary[f'{key}_std'] = float(np.std(values))
    return summary