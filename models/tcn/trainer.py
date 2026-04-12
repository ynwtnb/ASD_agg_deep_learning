"""
Training loop utilities for the TCN aggression prediction model.
"""

import json
import os
from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm=1.0):
    """
    Runs one full pass over the training DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
    loader : torch.utils.data.DataLoader
    optimizer : torch.optim.Optimizer
    criterion : torch.nn.BCEWithLogitsLoss
    device : torch.device
    max_grad_norm : float
        Gradient clipping norm. Prevents large updates from dilated conv layers
        in early training.

    Returns
    -------
    avg_loss : float
    accuracy : float
        Accuracy at 0.5 threshold, for training progress monitoring only.
    """
    model.train()
    total_loss = 0.0
    n_correct = 0
    n_total = 0

    for signals, labels in loader:
        signals = signals.to(device)
        labels_dev = labels.to(device)

        optimizer.zero_grad()
        logits = model(signals).squeeze(1)
        loss = criterion(logits, labels_dev.float())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if torch.isnan(grad_norm):
            print("  WARNING: NaN gradient norm detected, skipping batch")
            optimizer.zero_grad()
            continue
        optimizer.step()

        preds = (logits.sigmoid() > 0.5).long()
        total_loss += loss.item() * len(labels)
        n_correct += (preds == labels_dev).sum().item()
        n_total += len(labels)

    if n_total == 0:
        print("  WARNING: all batches skipped due to NaN gradients this epoch")
        return float('nan'), 0.0

    return total_loss / n_total, n_correct / n_total


def _validate(model, loader, criterion, device):
    """
    Single forward pass over a validation DataLoader.

    Returns val loss, accuracy at 0.5, and AUPRC in one pass to avoid
    iterating the loader multiple times per epoch.

    Parameters
    ----------
    model : torch.nn.Module
    loader : torch.utils.data.DataLoader
    criterion : torch.nn.BCEWithLogitsLoss
    device : torch.device

    Returns
    -------
    avg_loss : float
    accuracy : float
    auprc : float
    """
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            labels_dev = labels.to(device)
            logits = model(signals).squeeze(1)
            loss = criterion(logits, labels_dev.float())

            probs = logits.sigmoid()
            preds = (probs > 0.5).long()
            total_loss += loss.item() * len(labels)
            n_correct += (preds == labels_dev).sum().item()
            n_total += len(labels)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    try:
        auprc = float(average_precision_score(all_labels, all_probs))
    except ValueError:
        auprc = float('nan')

    return total_loss / n_total, n_correct / n_total, auprc


def _save_checkpoint(path, model, optimizer, scheduler, epoch,
                     best_val_auprc, best_state, epochs_no_improve, history):
    """
    Saves full training state to disk for resumption after job interruption.

    Parameters
    ----------
    path : str
        File path for the checkpoint.
    model : torch.nn.Module
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau or None
    epoch : int
        Zero-indexed epoch that just completed.
    best_val_auprc : float
    best_state : dict or None
    epochs_no_improve : int
    history : dict
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_val_auprc': best_val_auprc,
        'best_state': best_state,
        'epochs_no_improve': epochs_no_improve,
        'history': history,
    }, path)


def _load_checkpoint(path, model, optimizer, scheduler, device):
    """
    Loads full training state from a checkpoint file.

    Parameters
    ----------
    path : str
    model : torch.nn.Module
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau or None
    device : torch.device

    Returns
    -------
    epoch : int
        The epoch to resume from (next epoch index).
    best_val_auprc : float
    best_state : dict or None
    epochs_no_improve : int
    history : dict
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and ckpt['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    print(f"  resumed from checkpoint at epoch {ckpt['epoch'] + 1}")
    return (
        start_epoch,
        ckpt['best_val_auprc'],
        ckpt['best_state'],
        ckpt['epochs_no_improve'],
        ckpt['history'],
    )


def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          device, epochs, patience, save_path, max_grad_norm=1.0,
          warmup_epochs=5):
    """
    Full training loop with linear LR warmup, early stopping on val AUPRC,
    and per-epoch checkpointing for job resumption.

    For the first warmup_epochs epochs the learning rate ramps linearly from
    base_lr / 10 to base_lr. After warmup, ReduceLROnPlateau steps on negated
    val AUPRC each epoch.

    A full checkpoint (model, optimizer, scheduler, epoch, history) is saved
    after every epoch to save_path + '_checkpoint.pth'. If this file exists
    at the start of training, the loop resumes from it automatically.

    Parameters
    ----------
    model : torch.nn.Module
    train_loader : torch.utils.data.DataLoader
    val_loader : torch.utils.data.DataLoader
    criterion : torch.nn.BCEWithLogitsLoss
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
    device : torch.device
    epochs : int
    patience : int
    save_path : str
    max_grad_norm : float
    warmup_epochs : int
        Number of epochs for linear LR warmup from base_lr/10 to base_lr.

    Returns
    -------
    model : torch.nn.Module
        Loaded with best observed weights.
    history : dict
        Keys: train_loss, train_acc, val_loss, val_acc, val_auprc, lr.
    """
    base_lr = optimizer.param_groups[0]['lr']
    best_val_auprc = -1.0
    best_state = None
    epochs_no_improve = 0
    start_epoch = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auprc': [],
        'lr': [],
    }

    # resume from checkpoint if it exists
    checkpoint_path = save_path + '_checkpoint.pth' if save_path else None
    if checkpoint_path and os.path.isfile(checkpoint_path):
        start_epoch, best_val_auprc, best_state, epochs_no_improve, history = \
            _load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
        # restore base_lr from history if available
        if history['lr']:
            base_lr = max(history['lr'])

    for epoch in range(start_epoch, epochs):
        # linear warmup: ramp from base_lr/10 to base_lr
        if epoch < warmup_epochs:
            warmup_factor = 0.1 + 0.9 * (epoch / max(warmup_epochs, 1))
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * warmup_factor

        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_grad_norm
        )

        # if entire epoch was NaN, skip validation and count as no-improvement
        if np.isnan(train_loss):
            print(f"  epoch {epoch + 1:03d}/{epochs} | lr {current_lr:.2e} | SKIPPED (NaN)")
            epochs_no_improve += 1
            if checkpoint_path:
                _save_checkpoint(checkpoint_path, model, optimizer, scheduler,
                                 epoch, best_val_auprc, best_state,
                                 epochs_no_improve, history)
            if epochs_no_improve >= patience:
                print(f"  early stopping at epoch {epoch + 1}")
                break
            continue

        val_loss, val_acc, val_auprc = _validate(model, val_loader, criterion, device)

        # only step the plateau scheduler after warmup completes
        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step(-val_auprc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auprc'].append(val_auprc)
        history['lr'].append(current_lr)

        print(
            f"  epoch {epoch + 1:03d}/{epochs} | "
            f"lr {current_lr:.2e} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} auprc {val_auprc:.4f}"
        )

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
            if save_path is not None:
                torch.save(best_state, save_path + '_best.pth')
        else:
            epochs_no_improve += 1

        # save full checkpoint after every epoch for job resumption
        if checkpoint_path:
            _save_checkpoint(checkpoint_path, model, optimizer, scheduler,
                             epoch, best_val_auprc, best_state,
                             epochs_no_improve, history)

        if epochs_no_improve >= patience:
            print(f"  early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        with open(save_path + '_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    return model, history