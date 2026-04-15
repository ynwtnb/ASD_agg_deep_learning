"""
Optuna hyperparameter tuning worker for ShapeNet.

Each sbatch job runs this script as one independent Optuna worker.
Workers share a SQLite database to coordinate which trials to run.
The study is created automatically on first run (load_if_exists=True).

Usage:
    python optuna_worker.py \
        --data_path /path/to/data \
        --save_path /path/to/results \
        --storage sqlite:////abs/path/to/optuna.db \
        --study_name shapenet_tuning \
        --n_trials 1 \
        --split session \
        --cluster_num 50 \
        --cuda --gpu 0
"""

import os
import sys
import json
import argparse
import tempfile
import types

import numpy as np
import optuna
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../shared'))
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import load_dataset, run_split


# ── Search space ──────────────────────────────────────────────────────────────
# Edit this function to change which hyperparameters are tuned and their ranges.

def sample_params(trial):
    return {
        'batch_size':         trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
        'channels':           trial.suggest_int('channels', 10, 64),
        'depth':              trial.suggest_int('depth', 2, 8),
        'reduced_size':       trial.suggest_categorical('reduced_size', [80, 120, 160, 200]),
        'out_channels':       trial.suggest_categorical('out_channels', [80, 120, 160, 200]),
        'kernel_size':        trial.suggest_int('kernel_size', 2, 5),
        'epochs':             trial.suggest_int('epochs', 50, 200, step=50),
        'lr':                 trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'final_shapelet_num': trial.suggest_int('final_shapelet_num', 10, 50, step=10),
        'compared_length':    None,
    }


# ── Objective ─────────────────────────────────────────────────────────────────

def objective(trial, args, dataset):
    # ── Resume mode: reuse an existing (interrupted) trial directory ──────────
    # The interrupted trial's params are read from its trial_params.json so we
    # continue with identical hyperparameters.  Existing encoder / shapelet /
    # feature caches in that directory will be picked up automatically because
    # run_from_scratch=False → use_cache=True in _run_one_fold.
    #
    # Normal mode: create a fresh directory named after the new trial number.
    if args.resume_trial_dir:
        trial_save_path = args.resume_trial_dir
        run_from_scratch = False
        params_path = os.path.join(trial_save_path, 'trial_params.json')
        with open(params_path) as f:
            params = json.load(f)['params']
        print(f"[resume] Resuming from {trial_save_path}")
        print(f"[resume] Params: {params}")
    else:
        trial_save_path = os.path.join(args.save_path, f'trial_{trial.number}')
        run_from_scratch = True
        params = sample_params(trial)

    os.makedirs(trial_save_path, exist_ok=True)

    # Write params to a temp JSON so run_split can read it via fit_parameters
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(params, f)
        params_file = f.name

    try:
        # Build a minimal args namespace compatible with run_split
        fold_args = types.SimpleNamespace(
            split=args.split,
            n_splits=args.n_splits,
            val_prop=args.val_prop,
            stride=args.stride,
            cuda=args.cuda,
            gpu=args.gpu,
            seed=args.seed,
            max_discovery_samples=args.max_discovery_samples,
            load=False,
            run_from_scratch=run_from_scratch,
        )
        try:
            val_aurocs = run_split(
                fold_args, dataset, params_file,
                trial_save_path, args.cluster_num,
                run_config=None,
            )
        except torch.cuda.OutOfMemoryError:
            raise optuna.exceptions.TrialPruned("CUDA out of memory")
    finally:
        os.unlink(params_file)

    # Save trial params alongside results for easy reference
    with open(os.path.join(trial_save_path, 'trial_params.json'), 'w') as f:
        json.dump({'trial_number': trial.number, 'params': params}, f, indent=2)

    valid = [v for v in val_aurocs if not np.isnan(v)]
    if not valid:
        raise optuna.exceptions.TrialPruned("All val AUROC values are NaN")
    return float(np.mean(valid))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(description='Optuna worker for ShapeNet hyperparameter tuning')
    parser.add_argument('--data_path',   type=str, required=True)
    parser.add_argument('--save_path',   type=str, required=True)
    parser.add_argument('--storage',     type=str, required=True,
                        help='Optuna storage URL, e.g. sqlite:////abs/path/to/optuna.db')
    parser.add_argument('--study_name',  type=str, default='shapenet_tuning')
    parser.add_argument('--n_trials',    type=int, default=1,
                        help='number of trials this worker will run (default: 1)')
    parser.add_argument('--split',       type=str, default='session',
                        choices=['loso', 'kfold', 'session'])
    parser.add_argument('--n_splits',    type=int, default=5)
    parser.add_argument('--cluster_num', type=int, default=50)
    parser.add_argument('--val_prop',    type=float, default=0.2)
    parser.add_argument('--stride',      type=int, default=1,
                        help='stride for subsampling training data within each session. '
                             '1 = no subsampling (default: 1)')
    parser.add_argument('--cuda',        action='store_true')
    parser.add_argument('--gpu',         type=int, default=0)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--bin_size',    type=int, default=15)
    parser.add_argument('--num_observation_frames', type=int, default=12)
    parser.add_argument('--num_prediction_frames',  type=int, default=12)
    parser.add_argument('--max_discovery_samples',  type=int, default=500)
    parser.add_argument('--resume_trial_dir', type=str, default=None,
                        help='Path to an interrupted trial directory to resume from. '
                             'Params are read from trial_params.json inside that directory.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    os.makedirs(args.save_path, exist_ok=True)

    dataset = load_dataset(
        data_path=args.data_path,
        bin_size=args.bin_size,
        num_observation_frames=args.num_observation_frames,
        num_prediction_frames=args.num_prediction_frames,
    )

    # create_study with load_if_exists=True:
    # first worker creates the study, subsequent workers load it
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='maximize',  # maximise val AUROC
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, args, dataset),
        n_trials=args.n_trials,
    )

    print(f"\nBest trial:     #{study.best_trial.number}")
    print(f"Best val AUROC: {study.best_trial.value:.4f}")
    print(f"Best params:    {study.best_trial.params}")
