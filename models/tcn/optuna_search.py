"""
Optuna hyperparameter and architecture search for the TCN aggression model.

Uses a shared SQLite study so multiple SLURM workers can run in parallel.
Each worker runs 1 trial for crash isolation. TPE sampler coordinates across
workers via the shared database.

Pruning: MedianPruner kills trials whose val AUROC at any epoch falls below
the median of completed trials at that epoch. This cuts bad configurations
early and saves 50-70% of GPU time.

Search space covers:
    Architecture: channel widths, depth (n_blocks), kernel_size, readout
    Optimization: lr, optimizer (Adam/AdamW/SGD), lr schedule, batch_size
    Regularization: dropout, weight_decay, label_smoothing, pos_weight
    Data: train_stride (overlap reduction)

Usage:
    python optuna_search.py \
        --data_path /scratch/username/CBS_DATA_ASD_ONLY \
        --study_path experiments/results/tcn/optuna/study.db \
        --save_path experiments/results/tcn/optuna \
        --n_trials 1 \
        --cuda
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy

TCN_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.join(TCN_DIR, '..', '..', 'shared')
sys.path.insert(0, TCN_DIR)
sys.path.insert(0, SHARED_DIR)

from dataset import ASDAggressionDataset
from splitters import session_splits
from tcn import AggressionTCN
from evaluator import evaluate, find_optimal_threshold
from trainer import train_one_epoch, _validate
from pipeline import (
    NormSubset, compute_norm_stats, compute_pos_weight,
    make_val_split, subsample_train_subset, set_seed, SEED,
)

# suppress optuna's per-trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============= Architecture Definitions =============

# channel lists are keyed by n_blocks so that depth and width are coupled
# to maintain reasonable receptive fields and parameter counts

ARCH_PRESETS = {
    # 8-block architectures (RF with k=7: 3061)
    "8b_wide":   [64, 64, 128, 128, 256, 256, 256, 256],   # ~4M params
    "8b_mid":    [48, 48, 96, 96, 192, 192, 192, 192],      # ~2M params
    "8b_narrow": [32, 32, 64, 64, 128, 128, 128, 128],      # ~1M params
    # 6-block architectures (RF with k=7: 757, with k=13: 1513)
    "6b_wide":   [64, 64, 128, 128, 256, 256],              # ~2M params
    "6b_mid":    [48, 48, 96, 96, 192, 192],                 # ~1M params
    "6b_narrow": [32, 32, 64, 64, 128, 128],                 # ~0.5M params
}


# ============= Label Smoothing Loss =============

class LabelSmoothingBCE(nn.Module):
    """
    BCEWithLogitsLoss with label smoothing.

    Replaces hard 0/1 targets with smoothed targets to prevent
    overconfident predictions and reduce overfitting.

    Parameters
    ----------
    smoothing : float
        Amount of smoothing. 0.0 = no smoothing, 0.1 = targets become
        0.05 and 0.95 instead of 0 and 1.
    pos_weight : torch.Tensor or None
    """

    def __init__(self, smoothing=0.0, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


# ============= Pruning Callback =============

class OptunaPruneCallback:
    """
    Reports val AUROC to Optuna after each epoch for median pruning.

    Parameters
    ----------
    trial : optuna.Trial
    """

    def __init__(self, trial):
        self.trial = trial
        self.epoch = 0

    def __call__(self, val_auroc):
        self.trial.report(val_auroc, self.epoch)
        self.epoch += 1
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# ============= Training with Pruning =============

def train_with_pruning(model, train_loader, val_loader, criterion, optimizer,
                       scheduler, scheduler_type, device, epochs, patience,
                       max_grad_norm, warmup_epochs, prune_callback):
    """
    Training loop with Optuna pruning support.

    Parameters
    ----------
    model : torch.nn.Module
    train_loader : DataLoader
    val_loader : DataLoader
    criterion : nn.Module
    optimizer : torch.optim.Optimizer
    scheduler : lr scheduler or None
    scheduler_type : str
        'plateau' or 'cosine' -- determines how scheduler.step() is called.
    device : torch.device
    epochs : int
    patience : int
    max_grad_norm : float
    warmup_epochs : int
    prune_callback : OptunaPruneCallback

    Returns
    -------
    best_val_auroc : float
    model : torch.nn.Module with best weights loaded
    """
    base_lr = optimizer.param_groups[0]['lr']
    best_val_auroc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # linear warmup
        if epoch < warmup_epochs:
            warmup_factor = 0.1 + 0.9 * (epoch / max(warmup_epochs, 1))
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * warmup_factor

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_grad_norm
        )

        if np.isnan(train_loss):
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
            continue

        val_loss, val_acc, val_auprc, val_auroc = _validate(
            model, val_loader, criterion, device
        )

        # scheduler step depends on type
        if scheduler is not None and epoch >= warmup_epochs:
            if scheduler_type == 'plateau':
                scheduler.step(-val_auroc)
            elif scheduler_type == 'cosine':
                scheduler.step()

        # report val AUROC to Optuna for pruning
        prune_callback(val_auroc)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_auroc, model


# ============= Objective =============

def create_objective(dataset, device):
    """
    Returns an Optuna objective function closed over the dataset and device.

    Parameters
    ----------
    dataset : ASDAggressionDataset
    device : torch.device

    Returns
    -------
    objective : callable
    """
    # split once, reuse across all trials
    train_subset, test_subset = session_splits(dataset)

    def objective(trial):
        set_seed(SEED + trial.number)

        # ── architecture ────────────────────────────────────────────────
        arch_name = trial.suggest_categorical(
            "arch",
            ["8b_wide", "8b_mid", "8b_narrow", "6b_wide", "6b_mid", "6b_narrow"],
        )
        channel_list = ARCH_PRESETS[arch_name]

        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7, 9, 11])
        readout = trial.suggest_categorical("readout", ["mean", "last", "adaptive_max"])

        # ── optimization ────────────────────────────────────────────────
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
        lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        warmup_epochs = trial.suggest_int("warmup_epochs", 3, 15)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        lr_schedule = trial.suggest_categorical("lr_schedule", ["plateau", "cosine"])

        # ── regularization ──────────────────────────────────────────────
        dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.05)
        max_pos_weight = trial.suggest_float("max_pos_weight", 0.0, 10.0)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.15, step=0.05)

        # ── data ────────────────────────────────────────────────────────
        train_stride = trial.suggest_categorical("train_stride", [1, 4, 6])

        # fixed
        epochs = 50
        patience = 15
        max_grad_norm = 1.0

        # ── data preparation ────────────────────────────────────────────
        inner_train, inner_val = make_val_split(train_subset, val_prop=0.2)
        mean, std = compute_norm_stats(inner_train)

        if train_stride > 1:
            inner_train = subsample_train_subset(inner_train, stride=train_stride)

        train_norm = NormSubset(inner_train, mean, std)
        val_norm = NormSubset(inner_val, mean, std)
        test_norm = NormSubset(test_subset, mean, std)

        pin = device.type == 'cuda'
        train_loader = DataLoader(
            train_norm, batch_size=batch_size,
            shuffle=True, num_workers=0, pin_memory=pin,
        )
        val_loader = DataLoader(
            val_norm, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=pin,
        )
        test_loader = DataLoader(
            test_norm, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=pin,
        )

        # ── model ───────────────────────────────────────────────────────
        model = AggressionTCN(
            n_input_channels=10,
            channel_list=channel_list,
            kernel_size=kernel_size,
            dropout=dropout,
            readout=readout,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trial.set_user_attr("n_params", n_params)
        trial.set_user_attr("receptive_field", model.receptive_field)

        # ── loss ────────────────────────────────────────────────────────
        if max_pos_weight > 0:
            pw = compute_pos_weight(inner_train, max_pos_weight=max_pos_weight).to(device)
        else:
            pw = None

        if label_smoothing > 0:
            criterion = LabelSmoothingBCE(smoothing=label_smoothing, pos_weight=pw)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        # ── optimizer ───────────────────────────────────────────────────
        if optimizer_name == 'adam':
            optim = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay,
            )
        elif optimizer_name == 'adamw':
            optim = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay,
            )
        elif optimizer_name == 'sgd':
            optim = torch.optim.SGD(
                model.parameters(), lr=lr, weight_decay=weight_decay,
                momentum=0.9,
            )

        # ── lr schedule ─────────────────────────────────────────────────
        if lr_schedule == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode='min', patience=5, factor=0.5,
            )
        elif lr_schedule == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=epochs - warmup_epochs,
            )

        # ── train with pruning ──────────────────────────────────────────
        prune_cb = OptunaPruneCallback(trial)
        try:
            best_val_auroc, model = train_with_pruning(
                model, train_loader, val_loader, criterion, optim,
                scheduler, lr_schedule, device, epochs, patience,
                max_grad_norm, warmup_epochs, prune_cb,
            )
        except optuna.TrialPruned:
            raise

        # ── test evaluation ─────────────────────────────────────────────
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for signals, labels in val_loader:
                probs = model(signals.to(device)).squeeze(1).sigmoid()
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_probs_arr = np.array(val_probs)
        val_labels_arr = np.array(val_labels)

        if np.isnan(val_probs_arr).any() or val_labels_arr.sum() == 0:
            threshold = 0.5
        else:
            threshold = find_optimal_threshold(val_labels_arr, val_probs_arr)

        test_metrics = evaluate(model, test_loader, device, threshold=threshold)

        # store test metrics as trial attributes for analysis
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                trial.set_user_attr(f"test_{k}", v)
            elif hasattr(v, 'tolist'):
                trial.set_user_attr(f"test_{k}", v.tolist())

        print(
            f"  trial {trial.number:03d} | "
            f"arch={arch_name} k={kernel_size} ro={readout} "
            f"opt={optimizer_name} sched={lr_schedule} "
            f"lr={lr:.2e} do={dropout:.2f} wd={weight_decay:.2e} "
            f"pw={max_pos_weight:.1f} ls={label_smoothing:.2f} "
            f"stride={train_stride} bs={batch_size} "
            f"params={n_params:,} | "
            f"val_auroc={best_val_auroc:.4f} | "
            f"test_auprc={test_metrics['auprc']:.4f} "
            f"test_auroc={test_metrics['auc_roc']:.4f}"
        )

        return best_val_auroc

    return objective


# ============= CLI =============

def parse_arguments():
    """Parses command-line arguments for the Optuna TCN search."""
    parser = argparse.ArgumentParser(
        description='Optuna search for TCN aggression model'
    )
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--study_path', type=str, required=True,
                        help='path to SQLite study database')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--n_trials', type=int, default=1,
                        help='number of trials for this worker (default: 1 for crash isolation)')
    parser.add_argument('--bin_size', type=int, default=15)
    parser.add_argument('--num_observation_frames', type=int, default=12)
    parser.add_argument('--num_prediction_frames', type=int, default=12)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    args.data_path = os.path.normpath(args.data_path)
    slurm_job_id = int(os.environ.get("SLURM_JOB_ID", 0))
    worker_seed = SEED + slurm_job_id
    set_seed(worker_seed)

    if args.cuda and not torch.cuda.is_available():
        print("CUDA not available, proceeding on CPU")
        args.cuda = False

    device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')
    print(f"device: {device}")

    print("loading dataset...")
    dataset = ASDAggressionDataset(
        data_path=args.data_path,
        num_observation_frames=args.num_observation_frames,
        num_prediction_frames=args.num_prediction_frames,
        bin_size=args.bin_size,
        o_run_from_scratch=False,
    )
    print(f"dataset loaded: {len(dataset)} instances")

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.dirname(args.study_path), exist_ok=True)

    study_name = f"tcn_obs{args.num_observation_frames}_pred{args.num_prediction_frames}"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{args.study_path}",
        sampler=TPESampler(seed=SEED, n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        direction="maximize",
        load_if_exists=True,
    )

    objective = create_objective(dataset, device)

    print(f"starting {args.n_trials} trial(s) (existing: {len(study.trials)})...")
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    # ── summary ─────────────────────────────────────────────────────────
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\n=== study summary ===")
    print(f"  total: {len(study.trials)} | complete: {len(complete)} | pruned: {len(pruned)}")

    if complete:
        print(f"  best val AUROC: {study.best_value:.4f}")
        print(f"  best params:")
        for k, v in study.best_params.items():
            print(f"    {k}: {v}")

        best_attrs = study.best_trial.user_attrs
        if best_attrs:
            print(f"  best test metrics:")
            for k, v in sorted(best_attrs.items()):
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")

        # save best config as JSON for direct use in pipeline.py
        best = study.best_params
        best_config = {
            "n_input_channels": 10,
            "channel_list": ARCH_PRESETS[best["arch"]],
            "kernel_size": best["kernel_size"],
            "readout": best["readout"],
            "dropout": best["dropout"],
            "lr": best["lr"],
            "weight_decay": best["weight_decay"],
            "warmup_epochs": best["warmup_epochs"],
            "batch_size": best["batch_size"],
            "max_pos_weight": best["max_pos_weight"],
            "label_smoothing": best["label_smoothing"],
            "train_stride": best["train_stride"],
            "optimizer": best["optimizer"],
            "lr_schedule": best["lr_schedule"],
            "epochs": 50,
            "patience": 15,
            "max_grad_norm": 1.0,
        }
        config_path = os.path.join(args.save_path, 'best_config.json')
        with open(config_path, 'w') as f:
            json.dump(best_config, f, indent=4)
        print(f"\n  best config saved to {config_path}")

    # save full trial history
    trials_path = os.path.join(args.save_path, 'all_trials.json')
    trials_data = []
    for t in study.trials:
        trials_data.append({
            'number': t.number,
            'state': t.state.name,
            'value': t.value,
            'params': t.params,
            'user_attrs': t.user_attrs,
        })
    with open(trials_path, 'w') as f:
        json.dump(trials_data, f, indent=2)
    print(f"  all trials saved to {trials_path}")