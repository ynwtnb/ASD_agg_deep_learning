"""
Evaluation utilities for the TCN aggression prediction model.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def evaluate(model, loader, device, threshold=0.5):
    """
    Computes classification metrics on a DataLoader.

    Sensitivity and specificity are the primary clinical metrics: a model that
    never predicts aggression has high accuracy on imbalanced data but zero
    clinical value. AUC-ROC is threshold-independent and reported alongside
    fixed-threshold metrics.

    Parameters
    ----------
    model : torch.nn.Module
    loader : torch.utils.data.DataLoader
    device : torch.device
    threshold : float
        Sigmoid probability threshold for positive class prediction.

    Returns
    -------
    metrics : dict
        Keys: accuracy, f1_binary, f1_macro, auc_roc,
        sensitivity, specificity, confusion_matrix.
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            logits = model(signals).squeeze(1)      # (batch,)
            probs = logits.sigmoid()
            preds = (probs > threshold).long()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # single class present in this fold's test set
        auc_roc = float('nan')

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_binary': f1_score(all_labels, all_preds, average='binary', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'auc_roc': auc_roc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
    }


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
        'accuracy', 'f1_binary', 'f1_macro',
        'auc_roc', 'sensitivity', 'specificity',
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