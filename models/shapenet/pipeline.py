import os
import sys
import json
import math
import torch
import numpy as np
import argparse
import timeit

# Allow imports from shapenet models dir (must come before shared/ to avoid shadowing)
sys.path.insert(0, os.path.dirname(__file__))
# Allow imports from shared/
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../../shared'))

import wrappers
from dataset import ASDAggressionDataset
from splitters import loso_splits, kfold_participant_splits, session_splits


def load_dataset(data_path, bin_size, num_observation_frames, num_prediction_frames,
                 o_multiclass=False, o_run_from_scratch=False):
    """
    Loads the ASD aggression dataset wrapped in ASDAggressionDataset.
    """
    dataset = ASDAggressionDataset(
        data_path=data_path,
        bin_size=bin_size,
        num_observation_frames=num_observation_frames,
        num_prediction_frames=num_prediction_frames,
        o_multiclass=o_multiclass,
        o_run_from_scratch=o_run_from_scratch,
    )
    return dataset


def subset_to_numpy(subset):
    """
    Extracts numpy arrays (X, y) from a torch Subset of ASDAggressionDataset.

    Returns
    -------
    X : np.ndarray, shape (N, C, T)
    y : np.ndarray, shape (N,)
    """
    indices = np.asarray(subset.indices).astype(int)
    X = subset.dataset.instances[indices]
    y = subset.dataset.labels[indices]
    return X, y


def normalize(train, test):
    """
    Z-score normalises each channel independently using train-set statistics.
    Modifies arrays in-place and returns them.
    """
    nb_dims = train.shape[1]
    for j in range(nb_dims):
        mean = np.mean(train[:, j])
        var = np.var(train[:, j])
        std = math.sqrt(var) if var > 0 else 1.0
        train[:, j] = (train[:, j] - mean) / std
        test[:, j] = (test[:, j] - mean) / std
    return train, test


def fit_parameters(file, train, train_labels, test, test_labels, cuda, gpu,
                   save_path, cluster_num, save_memory=False):
    """
    Instantiates a CausalCNNEncoderClassifier from a JSON hyperparameter file,
    fits it on the training data, and returns the trained classifier.

    Parameters
    ----------
    file : str
        Path to JSON hyperparameter file.
    train : np.ndarray, shape (N, C, T)
    train_labels : np.ndarray, shape (N,)
    test : np.ndarray, shape (N, C, T)
    test_labels : np.ndarray, shape (N,)
    cuda : bool
    gpu : int
    save_path : str
        Directory prefix used by the classifier for shapelet files.
    cluster_num : int
        Number of clusters for shapelet discovery.
    save_memory : bool
    """
    classifier = wrappers.CausalCNNEncoderClassifier()

    with open(file, 'r') as hf:
        params = json.load(hf)

    params['in_channels'] = train.shape[1]  # number of signal channels
    params['cuda'] = cuda
    params['gpu'] = gpu
    classifier.set_params(**params)

    return classifier.fit(
        train, train_labels, test, test_labels,
        save_path, cluster_num,
        save_memory=save_memory, verbose=True
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='ShapeNet pipeline for ASD aggression dataset'
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to the dataset directory')
    parser.add_argument('--save_path', type=str, required=True,
                        help='path where models and results are saved')
    parser.add_argument('--hyper', type=str, required=True,
                        help='path to JSON file with hyperparameters')
    parser.add_argument('--bin_size', type=int, default=15,
                        help='bin size for data preprocessing (default: 10)')
    parser.add_argument('--num_observation_frames', type=int, default=12,
                        help='number of observation frames (default: 12)')
    parser.add_argument('--num_prediction_frames', type=int, default=12,
                        help='number of prediction frames (default: 12)')
    parser.add_argument('--cluster_num', type=int, default=10,
                        help='number of clusters for shapelet discovery (default: 10)')
    parser.add_argument('--split', type=str, default='loso',
                        choices=['loso', 'kfold', 'session'],
                        help='splitting strategy: loso | kfold | session (default: loso)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='number of folds for kfold split (default: 5)')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA if available')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--multiclass', action='store_true', default=False,
                        help='use multiclass labels instead of binary')
    parser.add_argument('--run_from_scratch', action='store_true', default=False,
                        help='re-run data extraction from scratch')
    parser.add_argument('--load', action='store_true', default=False,
                        help='load saved model(s) instead of training')

    print('parse arguments succeed !!!')
    return parser.parse_args()


if __name__ == '__main__':
    start = timeit.default_timer()
    args = parse_arguments()

    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
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

    # ── LOSO (leave-one-subject-out) ────────────────────────────────────────
    if args.split == 'loso':
        for test_pid, train_subset, test_subset in loso_splits(dataset):
            print(f"\n=== LOSO fold: test participant {test_pid} ===")
            train, train_labels = subset_to_numpy(train_subset)
            test, test_labels = subset_to_numpy(test_subset)
            train, test = normalize(train, test)
            print(f"  train={len(train)}, test={len(test)}")

            fold_save_path = os.path.join(args.save_path, f'pid_{test_pid}')
            os.makedirs(fold_save_path, exist_ok=True)
            prefix = os.path.join(fold_save_path, f'pid_{test_pid}')

            if not args.load:
                classifier = fit_parameters(
                    args.hyper, train, train_labels, test, test_labels,
                    args.cuda, args.gpu, prefix, args.cluster_num
                )
                classifier.save(prefix)
                with open(prefix + '_parameters.json', 'w') as fp:
                    json.dump(classifier.get_params(), fp)
            else:
                classifier = wrappers.CausalCNNEncoderClassifier()
                with open(prefix + '_parameters.json', 'r') as f:
                    hp_dict = json.load(f)
                hp_dict['cuda'] = args.cuda
                hp_dict['gpu'] = args.gpu
                classifier.set_params(**hp_dict)
                classifier.load(prefix)

    # ── K-Fold participant splits ────────────────────────────────────────────
    elif args.split == 'kfold':
        for fold, train_subset, test_subset in kfold_participant_splits(dataset, n_splits=args.n_splits):
            print(f"\n=== K-Fold: fold {fold} ===")
            train, train_labels = subset_to_numpy(train_subset)
            test, test_labels = subset_to_numpy(test_subset)
            train, test = normalize(train, test)
            print(f"  train={len(train)}, test={len(test)}")

            fold_save_path = os.path.join(args.save_path, f'fold_{fold}')
            os.makedirs(fold_save_path, exist_ok=True)
            prefix = os.path.join(fold_save_path, f'fold_{fold}')

            if not args.load:
                classifier = fit_parameters(
                    args.hyper, train, train_labels, test, test_labels,
                    args.cuda, args.gpu, prefix, args.cluster_num
                )
                classifier.save(prefix)
                with open(prefix + '_parameters.json', 'w') as fp:
                    json.dump(classifier.get_params(), fp)
            else:
                classifier = wrappers.CausalCNNEncoderClassifier()
                with open(prefix + '_parameters.json', 'r') as f:
                    hp_dict = json.load(f)
                hp_dict['cuda'] = args.cuda
                hp_dict['gpu'] = args.gpu
                classifier.set_params(**hp_dict)
                classifier.load(prefix)

    # ── Session-based split ─────────────────────────────────────────────────
    elif args.split == 'session':
        print("\n=== Session split ===")
        train_subset, test_subset = session_splits(dataset)
        train, train_labels = subset_to_numpy(train_subset)
        test, test_labels = subset_to_numpy(test_subset)
        train, test = normalize(train, test)
        print(f"  train={len(train)}, test={len(test)}")

        prefix = os.path.join(args.save_path, 'session_model')

        if not args.load:
            classifier = fit_parameters(
                args.hyper, train, train_labels, test, test_labels,
                args.cuda, args.gpu, prefix, args.cluster_num
            )
            classifier.save(prefix)
            with open(prefix + '_parameters.json', 'w') as fp:
                json.dump(classifier.get_params(), fp)
        else:
            classifier = wrappers.CausalCNNEncoderClassifier()
            with open(prefix + '_parameters.json', 'r') as f:
                hp_dict = json.load(f)
            hp_dict['cuda'] = args.cuda
            hp_dict['gpu'] = args.gpu
            classifier.set_params(**hp_dict)
            classifier.load(prefix)

    end = timeit.default_timer()
    print(f"\nAll time: {(end - start) / 60:.2f} minutes")
