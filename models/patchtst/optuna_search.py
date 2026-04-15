"""
Optuna hyperparameter search for PatchTST with pruning.

Each SLURM job runs 1 trial and reports to a shared SQLite study.
Pruning kills bad trials after a few epochs to save compute.

Usage (via submit script):
    bash scripts/submit_optuna_patchtst.sh 12

Usage (single trial, manual):
    python models/patchtst/optuna_search.py \
        --data_path /scratch/borasaniya.t/CBS_DATA_ASD_ONLY/ \
        --study_path experiments/results/patchtst/optuna/study.db \
        --save_path experiments/results/patchtst/optuna \
        --n_trials 1 --cuda --gpu 0
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../shared'))
sys.path.insert(0, os.path.dirname(__file__))

import optuna
from optuna.trial import TrialState
import torch
import wrappers
from dataset import ASDAggressionDataset
from splitters import session_splits
from losses.bce import get_loss_fn
import utils


def load_data(data_path):
    print("Loading dataset...")
    dataset = ASDAggressionDataset(
        data_path=data_path,
        bin_size=15,
        num_observation_frames=12,
        num_prediction_frames=12,
    )
    print(f"Dataset loaded: {len(dataset)} instances")

    train_subset, test_subset = session_splits(dataset)
    indices_train = np.asarray(train_subset.indices).astype(int)
    indices_test = np.asarray(test_subset.indices).astype(int)

    X_train = train_subset.dataset.instances[indices_train]
    y_train = train_subset.dataset.labels[indices_train]
    X_test = test_subset.dataset.instances[indices_test]
    y_test = test_subset.dataset.labels[indices_test]

    print(f"  train={len(X_train)}, test={len(X_test)}")
    return X_train, y_train, X_test, y_test


def objective(trial, X_train, y_train, X_test, y_test, save_path, cuda, gpu,
              prune_epochs=5, total_epochs=15):
    params = {
        'lr':           trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'dropout':      trial.suggest_float('dropout', 0.1, 0.5),
        'head_dropout': trial.suggest_float('head_dropout', 0.1, 0.5),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'batch_size':   trial.suggest_categorical('batch_size', [32, 64, 128]),
        'd_model':      trial.suggest_categorical('d_model', [64, 128, 256]),
        'n_layers':     trial.suggest_int('n_layers', 2, 4),
        'ffn_dim':      trial.suggest_categorical('ffn_dim', [128, 256, 512]),
        'pos_weight':   trial.suggest_float('pos_weight', 2.0, 10.0),
        'patch_len':    trial.suggest_categorical('patch_len', [32, 64, 128]),
        'patch_stride': trial.suggest_categorical('patch_stride', [16, 32, 64]),
        'use_focal':    False,
        'use_onecycle': False,
        'cuda':         cuda,
        'gpu':          gpu,
    }

    prefix = os.path.join(save_path, f'trial_{trial.number}', f'trial_{trial.number}')
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    # Build model and data loaders manually so we can prune mid-training
    from networks.patchtst import AggPatchTST, build_patchtst_config
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    device = torch.device(f'cuda:{gpu}' if cuda and torch.cuda.is_available() else 'cpu')

    cfg = build_patchtst_config(
        d_model=params['d_model'], n_heads=8,
        n_layers=params['n_layers'], ffn_dim=params['ffn_dim'],
        dropout=params['dropout'], head_dropout=params['head_dropout'],
        patch_len=params['patch_len'], patch_stride=params['patch_stride'],
    )
    model = AggPatchTST(config=cfg).to(device)

    X_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(y_train).float()
    Xt_t = torch.from_numpy(X_test).float()
    yt_t = torch.from_numpy(y_test).float()

    train_loader = DataLoader(TensorDataset(X_t, y_t),
                              batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(Xt_t, yt_t),
                            batch_size=params['batch_size'])

    pos_w = torch.tensor([params['pos_weight']], device=device)
    criterion = get_loss_fn(pos_weight=pos_w)
    optimizer = AdamW(model.parameters(), lr=params['lr'],
                      weight_decay=params['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    best_auroc = 0.0

    for epoch in range(total_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch).prediction_logits.squeeze(1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                logits = model(x_batch).prediction_logits.squeeze(1)
                all_logits.append(logits.cpu())
                all_labels.append(y_batch)
        metrics = utils.compute_metrics(
            torch.cat(all_logits), torch.cat(all_labels)
        )
        auroc = metrics['auroc']
        if auroc > best_auroc:
            best_auroc = auroc

        scheduler.step()

        # Report to Optuna and check for pruning
        trial.report(auroc, epoch)
        if epoch >= prune_epochs and trial.should_prune():
            print(f"  Trial {trial.number} pruned at epoch {epoch+1} (AUROC={auroc:.4f})")
            raise optuna.exceptions.TrialPruned()

        print(f"  Trial {trial.number} | Epoch {epoch+1:02d}/{total_epochs} | AUROC={auroc:.4f}")

    # Save best results
    result = {'auroc': best_auroc, 'params': params}
    with open(prefix + '_val_results.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  Trial {trial.number} done — Best AUROC={best_auroc:.4f}")
    return best_auroc


def parse_arguments():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter search for PatchTST')
    parser.add_argument('--data_path',  type=str, required=True)
    parser.add_argument('--study_path', type=str, required=True,
                        help='Path to SQLite DB, e.g. experiments/results/optuna/study.db')
    parser.add_argument('--save_path',  type=str, required=True)
    parser.add_argument('--n_trials',   type=int, default=1)
    parser.add_argument('--study_name', type=str, default='patchtst_aggression')
    parser.add_argument('--prune_epochs', type=int, default=5,
                        help='Start pruning after this many epochs (default: 5)')
    parser.add_argument('--total_epochs', type=int, default=15,
                        help='Total epochs per trial (default: 15)')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu',  type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.dirname(args.study_path), exist_ok=True)

    X_train, y_train, X_test, y_test = load_data(args.data_path)

    storage = f'sqlite:///{os.path.realpath(args.study_path)}'
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=args.prune_epochs)

    study = optuna.create_study(
        direction='maximize',
        storage=storage,
        study_name=args.study_name,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, X_test, y_test,
            args.save_path, args.cuda, args.gpu,
            prune_epochs=args.prune_epochs,
            total_epochs=args.total_epochs,
        ),
        n_trials=args.n_trials,
    )

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if completed:
        print(f"\nBest AUROC: {study.best_value:.4f}")
        print(f"Best params: {json.dumps(study.best_params, indent=2)}")
        best_path = os.path.join(args.save_path, 'best_params.json')
        with open(best_path, 'w') as f:
            json.dump({'auroc': study.best_value, 'params': study.best_params}, f, indent=2)
        print(f"Best params saved to {best_path}")
