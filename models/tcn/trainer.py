"""
Training loop utilities for the TCN aggression prediction model.
"""

import json
from copy import deepcopy

import torch


def _get_evaluator():
    """Lazily imports evaluate_val_auprc to avoid sys.path dependency at module load time."""
    from evaluator import evaluate_val_auprc
    return evaluate_val_auprc


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
        Maximum norm for gradient clipping. Prevents large gradient updates
        from dilated convolution layers in early training.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    model.train()
    total_loss = 0.0
    n_correct = 0
    n_total = 0

    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(signals).squeeze(1)          # (batch,)
        loss = criterion(logits, labels.float())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        preds = (logits.sigmoid() > 0.5).long()
        total_loss += loss.item() * len(labels)
        n_correct += (preds == labels).sum().item()
        n_total += len(labels)

    return total_loss / n_total, n_correct / n_total


def validate(model, loader, criterion, device):
    """
    Evaluates loss and accuracy on a DataLoader without gradient computation.

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
    """
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            labels = labels.to(device)

            logits = model(signals).squeeze(1)      # (batch,)
            loss = criterion(logits, labels.float())

            preds = (logits.sigmoid() > 0.5).long()
            total_loss += loss.item() * len(labels)
            n_correct += (preds == labels).sum().item()
            n_total += len(labels)

    return total_loss / n_total, n_correct / n_total


def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          device, epochs, patience, save_path, max_grad_norm=1.0):
    """
    Full training loop with early stopping on validation AUPRC.

    AUPRC is used as the early stopping signal rather than val loss because
    on imbalanced data val loss can decrease while AUPRC degrades if the model
    drifts toward predicting all-negative.

    Saves the best-epoch checkpoint to `save_path + '_best.pth'` and training
    history to `save_path + '_history.json'`.

    Parameters
    ----------
    model : torch.nn.Module
    train_loader : torch.utils.data.DataLoader
    val_loader : torch.utils.data.DataLoader
    criterion : torch.nn.BCEWithLogitsLoss
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler or None
        Expected to be ReduceLROnPlateau; stepped on val_auprc each epoch.
    device : torch.device
    epochs : int
    patience : int
    save_path : str
        File path prefix for checkpoint and history outputs.
    max_grad_norm : float
        Gradient clipping norm passed to train_one_epoch.

    Returns
    -------
    model : torch.nn.Module
        Loaded with best observed weights.
    history : dict
        Keys: 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auprc'.
    """
    best_val_auprc = -1.0
    best_state = None
    epochs_no_improve = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auprc': [],
    }

    evaluate_val_auprc = _get_evaluator()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_grad_norm
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_auprc = evaluate_val_auprc(model, val_loader, device)

        if scheduler is not None:
            # step on negative auprc so ReduceLROnPlateau minimises
            scheduler.step(-val_auprc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auprc'].append(val_auprc)

        print(
            f"  epoch {epoch + 1:03d}/{epochs} | "
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

        if epochs_no_improve >= patience:
            print(f"  early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        with open(save_path + '_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    return model, history