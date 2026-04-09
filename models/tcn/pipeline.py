"""
End-to-end supervised TCN training and evaluation pipeline for ASD aggression prediction.
"""

import argparse
import json
import os
import sys
import timeit
from math import trunc

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

TCN_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.join(TCN_DIR, '..', '..', 'shared')
sys.path.insert(0, TCN_DIR)
sys.path.insert(0, SHARED_DIR)

from dataset import ASDAggressionDataset
from splitters import loso_splits, kfold_participant_splits, session_splits
from tcn import AggressionTCN
from trainer import train
from evaluator import evaluate, summarize_metrics, find_optimal_threshold


# ============= Reproducibility =============

SEED = 42


def set_seed(seed):
    """Sets Python, NumPy, and PyTorch seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============= Normalization =============

class NormSubset(Subset):
    """
    Subset wrapper that applies per-channel z-score normalization in __getitem__.

    Stats are computed from the inner training split only and frozen at
    construction. Applying normalization here rather than in-place preserves
    the original dataset array, which is shared across all folds.

    Parameters
    ----------
    subset : torch.utils.data.Subset
    mean : np.ndarray of shape (C,)
        Per-channel mean computed from the inner training split.
    std : np.ndarray of shape (C,)
        Per-channel std computed from the inner training split.
    """

    def __init__(self, subset, mean, std):
        super().__init__(subset.dataset, subset.indices)
        self.mean = torch.tensor(mean, dtype=torch.float32).unsqueeze(1)  # (C, 1)
        self.std = torch.tensor(std, dtype=torch.float32).unsqueeze(1)    # (C, 1)

    def __getitem__(self, idx):
        signals, label = super().__getitem__(idx)
        return (signals - self.mean) / self.std, label


def compute_norm_stats(subset):
    """
    Computes per-channel mean and std over all instances and time steps.

    Parameters
    ----------
    subset : torch.utils.data.Subset

    Returns
    -------
    mean : np.ndarray of shape (C,)
    std : np.ndarray of shape (C,)
    """
    idx = np.asarray(subset.indices).astype(int)
    X = subset.dataset.instances[idx]   # (N, C, T)
    mean = X.mean(axis=(0, 2))
    std = X.std(axis=(0, 2))
    std[std == 0] = 1.0
    return mean, std


def compute_pos_weight(subset):
    """
    Computes BCE pos_weight = n_negative / n_positive from a training Subset.

    Parameters
    ----------
    subset : torch.utils.data.Subset

    Returns
    -------
    pos_weight : torch.Tensor of shape (1,)
    """
    idx = np.asarray(subset.indices).astype(int)
    labels = subset.dataset.labels[idx]
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)
    print(f"  class imbalance -- pos_weight: {weight.item():.2f}")
    return weight


# ============= Val Split =============

def make_val_split(train_subset, val_prop=0.2):
    """
    Chronological val split from a train Subset, respecting session boundaries.

    For each (subject, session) group, holds out the last val_prop fraction as
    val and removes the superposition overlap gap from the training tail to
    prevent temporal leakage. Sessions with fewer than 4 instances go entirely
    to inner training.

    Parameters
    ----------
    train_subset : torch.utils.data.Subset
    val_prop : float

    Returns
    -------
    inner_train : torch.utils.data.Subset
    inner_val : torch.utils.data.Subset
    """
    dataset = train_subset.dataset
    train_idx = np.asarray(train_subset.indices).astype(int)

    pids = dataset.get_participant_ids()[train_idx]
    session_ids = dataset.get_session_ids()[train_idx]
    sup = dataset.get_superposition_lists()[train_idx]  # (N, 2)

    inner_train_idx = []
    inner_val_idx = []

    for pid in np.unique(pids):
        pid_mask = pids == pid
        pid_idx = train_idx[pid_mask]
        pid_sessions = session_ids[pid_mask]
        pid_sup = sup[pid_mask]

        for sess in np.unique(pid_sessions):
            sess_mask = pid_sessions == sess
            sess_idx = pid_idx[sess_mask]
            sess_sup = pid_sup[sess_mask]

            n = len(sess_idx)
            if n < 4:
                inner_train_idx.extend(sess_idx.tolist())
                continue

            first_val = trunc(n * (1 - val_prop))
            n_overlap = int(sess_sup[first_val][0])
            last_train = first_val - n_overlap

            if last_train <= 0:
                inner_train_idx.extend(sess_idx.tolist())
                continue

            inner_train_idx.extend(sess_idx[:last_train].tolist())
            inner_val_idx.extend(sess_idx[first_val:].tolist())

    return Subset(dataset, inner_train_idx), Subset(dataset, inner_val_idx)


# ============= Model =============

def build_model(params, device):
    """
    Instantiates AggressionTCN from a hyperparameter dict and moves it to device.

    Parameters
    ----------
    params : dict
    device : torch.device

    Returns
    -------
    model : AggressionTCN
    """
    model = AggressionTCN(
        n_input_channels=params['n_input_channels'],
        channel_list=params['channel_list'],
        kernel_size=params['kernel_size'],
        dropout=params['dropout'],
        readout=params['readout'],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model parameters: {n_params:,}")
    print(f"  receptive field:  {model.receptive_field} samples")
    return model.to(device)


# ============= Fold Runner =============

def run_fold(train_subset, test_subset, params, device, save_path,
             val_prop=0.2, o_load=False):
    """
    Executes one train/evaluate cycle for a single fold.

    An inner chronological val split is created from train_subset via
    make_val_split. Normalization stats and pos_weight are derived from the
    inner training split only. Threshold for test evaluation is selected by
    maximising F1 on val predictions.

    Parameters
    ----------
    train_subset : torch.utils.data.Subset
    test_subset : torch.utils.data.Subset
    params : dict
    device : torch.device
    save_path : str
    val_prop : float
    o_load : bool

    Returns
    -------
    metrics : dict
    """
    inner_train, inner_val = make_val_split(train_subset, val_prop=val_prop)
    print(
        f"  inner split -- train: {len(inner_train)}"
        f"  val: {len(inner_val)}  test: {len(test_subset)} instances"
    )

    mean, std = compute_norm_stats(inner_train)
    train_norm = NormSubset(inner_train, mean, std)
    val_norm = NormSubset(inner_val, mean, std)
    test_norm = NormSubset(test_subset, mean, std)

    pin = device.type == 'cuda'
    train_loader = DataLoader(
        train_norm, batch_size=params['batch_size'],
        shuffle=True, num_workers=0, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_norm, batch_size=params['batch_size'],
        shuffle=False, num_workers=0, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_norm, batch_size=params['batch_size'],
        shuffle=False, num_workers=0, pin_memory=pin,
    )

    model = build_model(params, device)

    if o_load:
        model.load_state_dict(
            torch.load(save_path + '_best.pth', map_location=device)
        )
    else:
        pos_weight = compute_pos_weight(inner_train).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay'],
        )
        # negate auprc so ReduceLROnPlateau minimises the negated value
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5,
        )

        model, _ = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=params['epochs'],
            patience=params['patience'],
            save_path=save_path,
            max_grad_norm=params.get('max_grad_norm', 1.0),
        )
        torch.save(model.state_dict(), save_path + '_final.pth')

    # threshold selected on val, applied to test
    model.eval()
    val_probs, val_labels = [], []
    with torch.no_grad():
        for signals, labels in val_loader:
            probs = model(signals.to(device)).squeeze(1).sigmoid()
            val_probs.extend(probs.cpu().numpy())
            val_labels.extend(labels.numpy())
    val_probs_arr = np.array(val_probs)
    val_labels_arr = np.array(val_labels)
    print(f"  val set: {len(val_labels_arr)} instances, {int(val_labels_arr.sum())} positive")
    print(f"  val probs: min={val_probs_arr.min():.4f} max={val_probs_arr.max():.4f} nan={np.isnan(val_probs_arr).sum()}")

    if np.isnan(val_probs_arr).any():
        print("  WARNING: NaN in val probs — gradients likely exploded, check loss curve in history")
        threshold = 0.5
    elif val_labels_arr.sum() == 0:
        print("  WARNING: no positive instances in val set — threshold defaulting to 0.5")
        threshold = 0.5
    else:
        threshold = find_optimal_threshold(val_labels_arr, val_probs_arr)
    print(f"  optimal threshold (val F1): {threshold:.4f}")

    return evaluate(model, test_loader, device, threshold=threshold)


# ============= CLI =============

def parse_arguments():
    """Parses command-line arguments for the TCN pipeline."""
    parser = argparse.ArgumentParser(
        description='TCN pipeline for ASD aggression prediction'
    )
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--hyper', type=str, required=True)
    parser.add_argument('--bin_size', type=int, default=15)
    parser.add_argument('--num_observation_frames', type=int, default=12)
    parser.add_argument('--num_prediction_frames', type=int, default=12)
    parser.add_argument('--split', type=str, default='loso',
                        choices=['loso', 'kfold', 'session'])
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--val_prop', type=float, default=0.2)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--run_from_scratch', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=False)
    return parser.parse_args()


def _print_fold_metrics(metrics):
    """Prints a single-line per-fold summary to stdout."""
    print(
        f"  auprc={metrics['auprc']:.4f}  "
        f"auc={metrics['auc_roc']:.4f}  "
        f"f1={metrics['f1_binary']:.4f}  "
        f"sens={metrics['sensitivity']:.4f}  "
        f"spec={metrics['specificity']:.4f}  "
        f"thresh={metrics['threshold']:.4f}"
    )


def _save_metrics(all_metrics, path):
    """Saves per-fold metrics dict and aggregate summary to a JSON file."""
    serializable = {
        fold: {
            k: v.tolist() if hasattr(v, 'tolist') else v
            for k, v in m.items()
        }
        for fold, m in all_metrics.items()
    }
    summary = summarize_metrics(all_metrics)
    with open(path, 'w') as f:
        json.dump({'folds': serializable, 'summary': summary}, f, indent=2)


if __name__ == '__main__':
    start = timeit.default_timer()
    args = parse_arguments()

    set_seed(SEED)

    if args.cuda and not torch.cuda.is_available():
        print("CUDA not available, proceeding on CPU")
        args.cuda = False

    device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')
    print(f"device: {device}")

    with open(args.hyper, 'r') as f:
        params = json.load(f)

    print("loading dataset...")
    dataset = ASDAggressionDataset(
        data_path=args.data_path,
        num_observation_frames=args.num_observation_frames,
        num_prediction_frames=args.num_prediction_frames,
        bin_size=args.bin_size,
        o_run_from_scratch=args.run_from_scratch,
    )

    os.makedirs(args.save_path, exist_ok=True)
    all_metrics = {}

    # ── LOSO ──────────────────────────────────────────────────────────────────
    if args.split == 'loso':
        for test_pid, train_subset, test_subset in loso_splits(dataset):
            print(f"\n=== LOSO fold: test participant {test_pid} ===")
            fold_dir = os.path.join(args.save_path, f'pid_{test_pid}')
            os.makedirs(fold_dir, exist_ok=True)
            prefix = os.path.join(fold_dir, f'pid_{test_pid}')

            metrics = run_fold(
                train_subset, test_subset, params, device, prefix,
                val_prop=args.val_prop, o_load=args.load,
            )
            all_metrics[str(test_pid)] = metrics
            _print_fold_metrics(metrics)

    # ── K-Fold ────────────────────────────────────────────────────────────────
    elif args.split == 'kfold':
        for fold, train_subset, test_subset in kfold_participant_splits(
            dataset, n_splits=args.n_splits
        ):
            print(f"\n=== K-Fold: fold {fold} ===")
            fold_dir = os.path.join(args.save_path, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            prefix = os.path.join(fold_dir, f'fold_{fold}')

            metrics = run_fold(
                train_subset, test_subset, params, device, prefix,
                val_prop=args.val_prop, o_load=args.load,
            )
            all_metrics[f'fold_{fold}'] = metrics
            _print_fold_metrics(metrics)

    # ── Session ───────────────────────────────────────────────────────────────
    elif args.split == 'session':
        print("\n=== session split ===")
        train_subset, test_subset = session_splits(dataset)
        prefix = os.path.join(args.save_path, 'session_model')

        metrics = run_fold(
            train_subset, test_subset, params, device, prefix,
            val_prop=args.val_prop, o_load=args.load,
        )
        all_metrics['session'] = metrics
        _print_fold_metrics(metrics)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = summarize_metrics(all_metrics)
    print("\n=== aggregate summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    metrics_out = os.path.join(args.save_path, 'all_metrics.json')
    _save_metrics(all_metrics, metrics_out)
    print(f"\nmetrics saved to {metrics_out}")

    end = timeit.default_timer()
    print(f"total time: {(end - start) / 60:.2f} minutes")