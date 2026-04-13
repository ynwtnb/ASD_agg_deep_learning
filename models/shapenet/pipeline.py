import os
import sys
import json
import math
import random
import torch
import numpy as np
import argparse
import timeit

# Allow imports from shared/ and shapenet/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../shared'))
sys.path.insert(0, os.path.dirname(__file__))

import wrappers
from dataset import ASDAggressionDataset
from splitters import loso_splits, kfold_participant_splits, session_splits

torch.backends.cudnn.benchmark = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_run_config(args):
    """
    Build a dict of all arguments that affect training results.
    Used to validate whether cached intermediate files are still valid.
    """
    with open(args.hyper, 'r') as f:
        hyper_params = json.load(f)
    return {
        'bin_size': args.bin_size,
        'num_observation_frames': args.num_observation_frames,
        'num_prediction_frames': args.num_prediction_frames,
        'cluster_num': args.cluster_num,
        'multiclass': args.multiclass,
        'split': args.split,
        'n_splits': args.n_splits,
        'seed': args.seed,
        'hyper': hyper_params,
    }


def config_matches(prefix, run_config):
    """
    Returns True if a run_config saved at prefix matches the given run_config.
    """
    config_path = prefix + '_run_config.json'
    if not os.path.exists(config_path):
        return False
    with open(config_path, 'r') as f:
        saved = json.load(f)
    return saved == run_config


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
    print(f"labels shape: {dataset.labels.shape}")
    print(f"labels unique: {np.unique(dataset.labels)}")
    print(f"labels sample (first 10): {dataset.labels[:10]}")

    return dataset


def subset_to_numpy(subset):
    """
    Extracts numpy arrays (X, y, meta) from a torch Subset of ASDAggressionDataset.

    Returns
    -------
    X : np.ndarray, shape (N, C, T)
    y : np.ndarray, shape (N,)
    meta : dict with keys 'participant_ids', 'session_ids', 'superposition_lists'
    """
    indices = np.asarray(subset.indices).astype(int)
    X = subset.dataset.instances[indices]
    y = subset.dataset.labels[indices]
    meta = {
        'participant_ids': subset.dataset.participant_ids[indices],
        'session_ids': subset.dataset.session_ids[indices],
        'superposition_lists': subset.dataset.superposition_lists[indices],
    }
    return X, y, meta


def make_smoke_split(dataset, n_per_class):
    """
    Creates a tiny balanced train/test split directly from the full dataset,
    guaranteeing both pos and neg samples in both train and test sets.
    Takes 2*n_per_class per class for train and n_per_class per class for test
    (non-overlapping), falling back to available data if insufficient.
    """
    X_all = dataset.instances          # (N, C, T)
    y_all = dataset.labels             # (N,)
    if y_all.ndim > 1:
        y_all = y_all[:, 0]

    train_idx, test_idx = [], []
    for cls in np.unique(y_all):
        print(f"Number of unique classes in the dataset: {len(np.unique(y_all))}")

        cls_idx = np.where(y_all == cls)[0]
        # Always put at least 1 sample in train; share with test if data is scarce
        n_train = max(1, min(2 * n_per_class, len(cls_idx)))
        n_test = max(1, min(n_per_class, len(cls_idx)))
        train_idx.extend(cls_idx[:n_train].tolist())
        # Test samples start after train; wrap around to beginning if needed
        test_start = n_train if n_train < len(cls_idx) else 0
        test_idx.extend(cls_idx[test_start:test_start + n_test].tolist())

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    return (X_all[train_idx], y_all[train_idx],
            X_all[test_idx],  y_all[test_idx])


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
                   save_path, cluster_num, save_memory=False, override_epochs=None, seed=42, use_cache=False,
                   max_discovery_samples=500, test_meta=None):
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

    params['in_channels'] = 1  # ShapeNet flattens channels: each channel is treated as an independent univariate series
    params['cuda'] = cuda
    params['gpu'] = gpu
    params['seed'] = seed
    if override_epochs is not None:
        params['epochs'] = override_epochs
    classifier.set_params(**params)

    return classifier.fit(
        train, train_labels, test, test_labels,
        save_path, cluster_num,
        save_memory=save_memory, verbose=True, use_cache=use_cache,
        max_discovery_samples=max_discovery_samples,
        test_meta=test_meta,
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
    parser.add_argument('--smoke_test', action='store_true', default=False,
                        help='quick pipeline check: use only a tiny balanced subset per fold')
    parser.add_argument('--smoke_test_n', type=int, default=5,
                        help='samples per class when --smoke_test is set (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility (default: 42)')
    parser.add_argument('--max_discovery_samples', type=int, default=500,
                        help='max training samples for shapelet discovery (stratified subsample). '
                             'Lower if OOM during discovery. With T~2880 D=10, 500 needs ~29 GB. (default: 500)')

    print('parse arguments succeed !!!')
    return parser.parse_args()


if __name__ == '__main__':
    start = timeit.default_timer()
    args = parse_arguments()

    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        args.cuda = False

    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")

    t0 = timeit.default_timer()
    print("Loading dataset...")
    dataset = load_dataset(
        data_path=args.data_path,
        bin_size=args.bin_size,
        num_observation_frames=args.num_observation_frames,
        num_prediction_frames=args.num_prediction_frames,
        o_multiclass=args.multiclass,
        o_run_from_scratch=args.run_from_scratch,
    )
    print(f"[timing] dataset load: {(timeit.default_timer()-t0)/60:.3f} min | {len(dataset)} instances")

    os.makedirs(args.save_path, exist_ok=True)

    # Compute once; used by all fold branches to validate intermediate caches.
    run_config = make_run_config(args)

    # ── Smoke test: bypass fold logic, sample directly from full dataset ─────
    if args.smoke_test:
        print(f"\n=== SMOKE TEST ({args.smoke_test_n} samples/class for train, {args.smoke_test_n} for test) ===")

        t0 = timeit.default_timer()
        train, train_labels, test, test_labels = make_smoke_split(dataset, args.smoke_test_n)
        print(f"[timing] smoke split: {(timeit.default_timer()-t0)/60:.3f} min | train={len(train)}, test={len(test)}")

        t0 = timeit.default_timer()
        train, test = normalize(train, test)
        print(f"[timing] normalize: {(timeit.default_timer()-t0)/60:.3f} min")

        prefix = os.path.join(args.save_path, 'smoke_test')

        t0 = timeit.default_timer()
        classifier = fit_parameters(
            args.hyper, train, train_labels, test, test_labels,
            args.cuda, args.gpu, prefix, args.cluster_num,
            override_epochs=1, seed=args.seed,
            max_discovery_samples=args.max_discovery_samples,
        )
        print(f"[timing] fit_parameters (encoder+discovery+transform+svm): {(timeit.default_timer()-t0)/60:.3f} min")
        print("Smoke test passed.")

    # ── LOSO (leave-one-subject-out) ────────────────────────────────────────
    elif args.split == 'loso':
        for test_pid, train_subset, test_subset in loso_splits(dataset):
            print(f"\n=== LOSO fold: test participant {test_pid} ===")
            train, train_labels, _ = subset_to_numpy(train_subset)
            test, test_labels, test_meta = subset_to_numpy(test_subset)
            train, test = normalize(train, test)
            print(f"  train={len(train)}, test={len(test)}")

            fold_save_path = os.path.join(args.save_path, f'pid_{test_pid}')
            os.makedirs(fold_save_path, exist_ok=True)
            prefix = os.path.join(fold_save_path, f'pid_{test_pid}')

            if not args.load:
                use_cache = (not args.run_from_scratch) and config_matches(prefix, run_config)
                if not use_cache:
                    # Write-ahead: save config before training so partial results
                    # can be resumed on the next run with the same args.
                    with open(prefix + '_run_config.json', 'w') as fp:
                        json.dump(run_config, fp)
                classifier = fit_parameters(
                    args.hyper, train, train_labels, test, test_labels,
                    args.cuda, args.gpu, prefix, args.cluster_num,
                    seed=args.seed, use_cache=use_cache,
                    max_discovery_samples=args.max_discovery_samples,
                    test_meta=test_meta,
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
            train, train_labels, _ = subset_to_numpy(train_subset)
            test, test_labels, test_meta = subset_to_numpy(test_subset)
            train, test = normalize(train, test)
            print(f"  train={len(train)}, test={len(test)}")

            fold_save_path = os.path.join(args.save_path, f'fold_{fold}')
            os.makedirs(fold_save_path, exist_ok=True)
            prefix = os.path.join(fold_save_path, f'fold_{fold}')

            if not args.load:
                use_cache = (not args.run_from_scratch) and config_matches(prefix, run_config)
                if not use_cache:
                    with open(prefix + '_run_config.json', 'w') as fp:
                        json.dump(run_config, fp)
                classifier = fit_parameters(
                    args.hyper, train, train_labels, test, test_labels,
                    args.cuda, args.gpu, prefix, args.cluster_num,
                    seed=args.seed, use_cache=use_cache,
                    max_discovery_samples=args.max_discovery_samples,
                    test_meta=test_meta,
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
        train, train_labels, _ = subset_to_numpy(train_subset)
        test, test_labels, test_meta = subset_to_numpy(test_subset)
        train, test = normalize(train, test)
        print(f"  train={len(train)}, test={len(test)}")

        prefix = os.path.join(args.save_path, 'session_model')

        if not args.load:
            use_cache = (not args.run_from_scratch) and config_matches(prefix, run_config)
            if not use_cache:
                with open(prefix + '_run_config.json', 'w') as fp:
                    json.dump(run_config, fp)
            classifier = fit_parameters(
                args.hyper, train, train_labels, test, test_labels,
                args.cuda, args.gpu, prefix, args.cluster_num,
                seed=args.seed, use_cache=use_cache,
                max_discovery_samples=args.max_discovery_samples,
                test_meta=test_meta,
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
