import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score


# Metrics
def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Computes AUROC, F1, and AUPRC from raw logits and binary labels.

    Args:
        logits : torch.Tensor [N]  — raw (pre-sigmoid) model output
        labels : torch.Tensor [N]  — binary float labels {0.0, 1.0}
    Returns:
        dict with keys 'auroc', 'f1', 'auprc'
    """
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)
    labels = labels.numpy().astype(int)

    # Guard against folds where only one class is present
    if len(np.unique(labels)) < 2:
        return {'auroc': float('nan'), 'f1': float('nan'), 'auprc': float('nan')}

    return {
        'auroc': roc_auc_score(labels, probs),
        'f1':    f1_score(labels, preds, zero_division=0),
        'auprc': average_precision_score(labels, probs),
    }


# Checkpoint I/O
def save_checkpoint(path: str, model, optimizer, epoch: int, val_loss: float):
    """
    Saves model + optimizer state to disk.
    Called by wrappers.py whenever val_loss improves.
    """
    torch.save({
        'epoch':     epoch,
        'val_loss':  val_loss,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)


def load_checkpoint(path: str, model, optimizer=None):
    """
    Loads model state from disk.
    Returns (epoch, val_loss) so training can resume from the right point.
    """
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    print(
        f"Resumed from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    return ckpt['epoch'], ckpt['val_loss']
