"""
Training pipeline for PatchTST aggression model.

Usage:
    python pipeline.py \
        -data_path /path/to/data \
        -save_path /path/to/output \
        -hyper default_parameters.json \
        -split loso \
        -cuda
"""

from splitters import loso_splits, kfold_participant_splits, session_splits
from dataset import ASDAggressionDataset
import wrappers
import os
import sys
import json
import timeit
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../shared'))


# Dataset loader
def load_dataset(data_path, bin_size, num_observation_frames,
                 num_prediction_frames, o_multiclass=False,
                 o_run_from_scratch=False):
    return ASDAggressionDataset(
        data_path=data_path,
        bin_size=bin_size,
        num_observation_frames=num_observation_frames,
        num_prediction_frames=num_prediction_frames,
        o_multiclass=o_multiclass,
        o_run_from_scratch=o_run_from_scratch,
    )


def subset_to_numpy(subset):
    """
    Returns X [N, 10, 2880] and y [N] as numpy arrays.
    Identical to shapenet's subset_to_numpy.
    """
    indices = np.asarray(subset.indices).astype(int)
    X = subset.dataset.instances[indices]   # [N, 10, 2880]
    y = subset.dataset.labels[indices]      # [N]
    return X, y


def fit_parameters(hyper_file, train, train_labels, test, test_labels,
                   cuda, gpu, save_path):
    """
    Reads hyperparams from JSON, builds a PatchTSTClassifier, fits it.
    """
    classifier = wrappers.PatchTSTClassifier()

    with open(hyper_file, 'r') as f:
        params = json.load(f)

    params['cuda'] = cuda
    params['gpu'] = gpu
    classifier.set_params(**params)

    return classifier.fit(train, train_labels, test, test_labels, save_path)


# Argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='PatchTST pipeline for ASD aggression dataset'
    )
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--hyper',     type=str,
                        default='default_parameters.json')
    parser.add_argument('--bin_size',  type=int, default=15)
    parser.add_argument('--num_observation_frames', type=int, default=12)
    parser.add_argument('--num_prediction_frames',  type=int, default=12)
    parser.add_argument('--split', type=str, default='loso',
                        choices=['loso', 'kfold', 'session'])
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu',  type=int, default=0)
    parser.add_argument('--multiclass',
                        action='store_true', default=False)
    parser.add_argument('--run_from_scratch',
                        action='store_true', default=False)
    parser.add_argument(
        '--load',              action='store_true', default=False)
    print('parse arguments succeed !!!')
    return parser.parse_args()


# Main
if __name__ == '__main__':
    start = timeit.default_timer()
    args = parse_arguments()

    if args.cuda and not torch.cuda.is_available():
        print("CUDA not available, proceeding on CPU...")
        args.cuda = False

    print("Loading dataset...")
    dataset = load_dataset(
        data_path=args.data_path,
        bin_size=args.bin_size,
        num_observation_frames=args.num_observation_frames,
        num_prediction_frames=args.num_prediction_frames,
        o_multiclass=args.multiclass,
        o_run_from_scratch=args.run_from_scratch,
    )
    print(f"Dataset loaded: {len(dataset)} instances")
    os.makedirs(args.save_path, exist_ok=True)

    # LOSO
    if args.split == 'loso':
        for test_pid, train_subset, test_subset in loso_splits(dataset):
            print(f"\n=== LOSO fold: test participant {test_pid} ===")
            train, train_labels = subset_to_numpy(train_subset)
            test,  test_labels = subset_to_numpy(test_subset)
            print(f"  train={len(train)}, test={len(test)}")

            fold_save = os.path.join(args.save_path, f'pid_{test_pid}')
            os.makedirs(fold_save, exist_ok=True)
            prefix = os.path.join(fold_save, f'pid_{test_pid}')

            if not args.load:
                classifier = fit_parameters(
                    args.hyper, train, train_labels, test, test_labels,
                    args.cuda, args.gpu, prefix
                )
                classifier.save(prefix)
                with open(prefix + '_parameters.json', 'w') as fp:
                    json.dump(classifier.get_params(), fp)
                results_path = prefix + '_val_results.json'
                if os.path.exists(results_path):
                    with open(results_path) as f:
                        r = json.load(f)
                    print(f"  Best: epoch={r['epoch']} AUROC={r['auroc']:.4f} F1={r['f1']:.4f} AUPRC={r['auprc']:.4f}")
            else:
                classifier = wrappers.PatchTSTClassifier()
                with open(prefix + '_parameters.json', 'r') as f:
                    hp = json.load(f)
                hp['cuda'] = args.cuda
                hp['gpu'] = args.gpu
                classifier.set_params(**hp)
                classifier.load(prefix)

    #  K-Fold
    elif args.split == 'kfold':
        for fold, train_subset, test_subset in kfold_participant_splits(
                dataset, n_splits=args.n_splits):
            print(f"\n=== K-Fold: fold {fold} ===")
            train, train_labels = subset_to_numpy(train_subset)
            test,  test_labels = subset_to_numpy(test_subset)
            print(f"  train={len(train)}, test={len(test)}")

            fold_save = os.path.join(args.save_path, f'fold_{fold}')
            os.makedirs(fold_save, exist_ok=True)
            prefix = os.path.join(fold_save, f'fold_{fold}')

            if not args.load:
                classifier = fit_parameters(
                    args.hyper, train, train_labels, test, test_labels,
                    args.cuda, args.gpu, prefix
                )
                classifier.save(prefix)
                with open(prefix + '_parameters.json', 'w') as fp:
                    json.dump(classifier.get_params(), fp)
                results_path = prefix + '_val_results.json'
                if os.path.exists(results_path):
                    with open(results_path) as f:
                        r = json.load(f)
                    print(f"  Best: epoch={r['epoch']} AUROC={r['auroc']:.4f} F1={r['f1']:.4f} AUPRC={r['auprc']:.4f}")
            else:
                classifier = wrappers.PatchTSTClassifier()
                with open(prefix + '_parameters.json', 'r') as f:
                    hp = json.load(f)
                hp['cuda'] = args.cuda
                hp['gpu'] = args.gpu
                classifier.set_params(**hp)
                classifier.load(prefix)

    # Session
    elif args.split == 'session':
        print("\n=== Session split ===")
        train_subset, test_subset = session_splits(dataset)
        train, train_labels = subset_to_numpy(train_subset)
        test,  test_labels = subset_to_numpy(test_subset)
        print(f"  train={len(train)}, test={len(test)}")

        prefix = os.path.join(args.save_path, 'session_model')

        if not args.load:
            classifier = fit_parameters(
                args.hyper, train, train_labels, test, test_labels,
                args.cuda, args.gpu, prefix
            )
            classifier.save(prefix)
            with open(prefix + '_parameters.json', 'w') as fp:
                json.dump(classifier.get_params(), fp)
            results_path = prefix + '_val_results.json'
            if os.path.exists(results_path):
                with open(results_path) as f:
                    r = json.load(f)
                print(f"  Best: epoch={r['epoch']} AUROC={r['auroc']:.4f} F1={r['f1']:.4f} AUPRC={r['auprc']:.4f}")
        else:
            classifier = wrappers.PatchTSTClassifier()
            with open(prefix + '_parameters.json', 'r') as f:
                hp = json.load(f)
            hp['cuda'] = args.cuda
            hp['gpu'] = args.gpu
            classifier.set_params(**hp)
            classifier.load(prefix)

    end = timeit.default_timer()
    print(f"\nAll time: {(end - start) / 60:.2f} minutes")
