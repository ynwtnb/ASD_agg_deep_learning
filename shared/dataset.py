import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

from data_extraction import data_preprocess

class ASDAggressionDataset(Dataset):
    """
    Flat dataset of 3-min instances.
    Each instance has signals (C, T) tensor [channels x time] with an associated label (0 or 1 if binary classification).
    """

    def __init__(self, data_path, bin_size, num_observation_frames, num_prediction_frames, 
                o_multiclass=False, o_run_from_scratch=False):
        """
        Parameters
        ----------
        data_path : str
            The path to the data directory.
        bin_size : int
            The size of the bins to use for the data.
        num_observation_frames : int
            The number of observation frames to include in the data.
        num_prediction_frames : int
            The number of prediction frames to include in the data.
        o_multiclass : bool
            Whether to include the multiclass labels in the data.
        o_run_from_scratch : bool
            Whether to run the data extraction from scratch.
        """
        self.data_path = data_path
        self.bin_size = bin_size
        self.num_observation_frames = num_observation_frames
        self.num_prediction_frames = num_prediction_frames
        self.o_multiclass = o_multiclass
        self.o_run_from_scratch = o_run_from_scratch

        self.dict_of_instances_arrays, self.dict_of_labels_arrays, self.dict_of_session_id_arrays, self.id_blacklist, self.uid_dict, \
            self.dict_of_superposition_lists, self.feat_col_names, self.dict_of_session_dfs = data_preprocess(
                data_path=data_path, num_observation_frames=num_observation_frames, num_prediction_frames=num_prediction_frames,
                o_run_from_scratch=o_run_from_scratch, o_return_list_of_sessions=False,  
                bin_size=bin_size, o_multiclass=o_multiclass, 
                print_progress=False,
            )
        
        # Flatten all arrays, iterating over a shared key set to guarantee alignment.
        # Only keep participants that have data in ALL four dicts.
        valid_pids = (
            self.dict_of_instances_arrays.keys()
            & self.dict_of_labels_arrays.keys()
            & self.dict_of_session_id_arrays.keys()
            & self.dict_of_superposition_lists.keys()
        )

        n_channels = len(self.feat_col_names)
        all_instances, all_labels, all_session_ids, all_superposition, all_pids = [], [], [], [], []
        for pid in sorted(valid_pids):
            inst = self.dict_of_instances_arrays[pid]
            lbl = self.dict_of_labels_arrays[pid]      # (n,) or (n, pred_bins)
            sid = self.dict_of_session_id_arrays[pid]  # (n,)
            sup = self.dict_of_superposition_lists[pid]

            n = len(lbl)
            if n == 0:
                continue

            # Older cached data may store instances as 2D (n*C, T) instead of 3D (n, C, T).
            if inst.ndim == 2:
                inst = inst.reshape(n, n_channels, -1)

            # Older cached data may store superposition as a flat list [p0,f0,p1,f1,...] of length 2n.
            sup_arr = np.array(sup)
            if sup_arr.ndim == 1 and len(sup_arr) == 2 * n:
                sup_arr = sup_arr.reshape(n, 2)

            assert inst.shape[0] == n and sid.shape[0] == n and len(sup_arr) == n, \
                f"Participant {pid}: mismatched lengths — instances={inst.shape[0]}, labels={n}, " \
                f"session_ids={sid.shape[0]}, superposition={len(sup_arr)}"

            all_instances.append(inst)
            all_labels.append(lbl)
            all_session_ids.append(sid)
            all_superposition.append(sup_arr)  # (n, 2): [past_overlap, future_overlap]
            all_pids.append(np.full(n, pid))

        self.instances = np.concatenate(all_instances, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        self.session_ids = np.concatenate(all_session_ids, axis=0)
        self.superposition_lists = np.concatenate(all_superposition, axis=0)
        self.participant_ids = np.concatenate(all_pids, axis=0)
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        signals = torch.tensor(self.instances[idx], dtype=torch.float32)  # (C, T)
        label   = torch.tensor(self.labels[idx],   dtype=torch.long)

        return signals, label

    # ── Convenience accessors for splitters ──────────────────────────────
    def get_participant_ids(self) -> np.ndarray:
        return self.participant_ids

    def get_session_ids(self) -> np.ndarray:
        return self.session_ids
    
    def get_superposition_lists(self) -> list:
        return self.superposition_lists

    def unique_participants(self) -> list:
        return sorted(set(self.participant_ids))

    def unique_sessions(self) -> list:
        return sorted(set(self.session_ids))