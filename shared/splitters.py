import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from math import trunc

from dataset import ASDAggressionDataset

def loso_splits(dataset: ASDAggressionDataset):
    """
    Yields (train_subset, test_subset) for each participant as test fold.
    Val is None here — add an inner loop or heldout session if you need val.
    """
    pids = dataset.get_participant_ids()
    all_idx = np.arange(len(dataset))

    for test_pid in dataset.unique_participants():
        test_mask = pids == test_pid
        train_mask = ~test_mask

        train_idx = all_idx[train_mask]
        test_idx = all_idx[test_mask]

        yield (
            test_pid,
            Subset(dataset, train_idx),
            Subset(dataset, test_idx),
        )

def session_splits(dataset: ASDAggressionDataset, test_prop=0.2):
    """
    Split each session into train and test sets.
    test_prop: proportion of the session to hold out for testing.
    """
    pids = dataset.unique_participants()
    superposition_lists = dataset.get_superposition_lists()
    session_ids = dataset.get_session_ids()
    instances = dataset.instances
    all_idx = np.arange(len(dataset))

    # use lists to avoid float64 indices from np.concatenate with empty array
    train_idx = []
    test_idx = []

    for pid in pids:
        pid_mask = dataset.get_participant_ids() == pid
        pid_session_ids = session_ids[pid_mask]
        pid_instances = instances[pid_mask]
        pid_superposition_lists = superposition_lists[pid_mask]

        for session_id in set(pid_session_ids):
            session_mask = pid_session_ids == session_id
            session_superposition_lists = pid_superposition_lists[session_mask]
            session_all_idx = all_idx[pid_mask][session_mask]

            n_samples = len(pid_instances[session_mask])
            first_test_sample_idx = trunc(n_samples * (1 - test_prop))
            n_overlapping_samples = int(session_superposition_lists[first_test_sample_idx][0])
            last_training_sample_idx = first_test_sample_idx - n_overlapping_samples

            # skip: session too short to produce a valid train segment after gap removal
            if last_training_sample_idx <= 0:
                continue

            train_idx.extend(session_all_idx[:last_training_sample_idx].tolist())
            test_idx.extend(session_all_idx[first_test_sample_idx:].tolist())

    return Subset(dataset, train_idx), Subset(dataset, test_idx)

def kfold_participant_splits(dataset: ASDAggressionDataset, n_splits=5, seed=42):
    """
    K-fold where each fold holds out a disjoint subset of participants.
    Instances from the same participant stay together.
    """
    participants = np.array(dataset.unique_participants())
    pids = dataset.get_participant_ids()
    all_idx = np.arange(len(dataset))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_p_idx, test_p_idx) in enumerate(kf.split(participants)):
        train_pids = participants[train_p_idx]
        test_pids  = participants[test_p_idx]

        train_idx = all_idx[np.isin(pids, train_pids)]
        test_idx  = all_idx[np.isin(pids, test_pids)]

        yield fold, Subset(dataset, train_idx), Subset(dataset, test_idx)