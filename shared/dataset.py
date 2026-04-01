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
        
        # Flatten the dict_of_instances_arrays and dict_of_labels_arrays
        self.instances = np.concatenate([np.concatenate(instances) for instances in self.dict_of_instances_arrays.values()], axis=0)
        self.labels = np.concatenate([np.concatenate(labels) for labels in self.dict_of_labels_arrays.values()], axis=0)
        self.session_ids = np.concatenate([np.concatenate(session_ids) for session_ids in self.dict_of_session_id_arrays.values()], axis=0)
        self.superposition_lists = [np.concatenate(superposition_lists) for superposition_lists in self.dict_of_superposition_lists.values()]
        self.participant_ids = np.concatenate([np.full(len(inst), pid) for pid, inst in self.dict_of_instances_arrays.items()], axis=0)
        
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