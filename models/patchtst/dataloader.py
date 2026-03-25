"""
PatchTST DataLoader
- load .bin file
- split by subject id (train/val/test)
- transpose to code format [batch, seq_len, channels]
- return pytorch dataloader obj
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SIGNAL_COLS = ['BVP', 'EDA', 'ACC_X', 'ACC_Y', 'ACC_Z', 
               'Magnitude', 'HR', 'RMSSD', 'PHASIC', 'TONIC']

N_CHANNELS = len(SIGNAL_COLS)   # 10
TARGET_FS = 16                 # Hz
BIN_SIZE = 15                 # sec
N_OBS_BINS = 12                 # 3 mins
SEQ_LEN = N_OBS_BINS * TARGET_FS * BIN_SIZE  # 12 × 16 × 15 = 2880


class AggDatasetPatchTST(Dataset):
	"""
	Dataset wrapping
	- handle transposition via __getitem__

	Parameters
	dict_of_instances : dict
			{subject_id: np.ndarray of shape [n_instances, 10, 2880]}
			or {subject_id: list of arrays} if o_return_list_of_sessions=True
	dict_of_labels : dict
			{subject_id: np.ndarray of shape [n_instances]}
	subject_ids : list
			Subjects to include in this split (train/val/test)
	"""

	def __init__(self, dict_of_instances, dict_of_labels, subject_ids):
		self.instances = []
		self.labels = []

		for sid in subject_ids:
			if sid not in dict_of_instances:
				print(f"Subject {sid} not found in data, skipping.")
				continue

			x = dict_of_instances[sid]
			y = dict_of_labels[sid]

			# Handle list-of-sessions format
			if isinstance(x, list):
				if len(x) == 0:
					continue
				x = np.concatenate(x, axis=0)   # [total_instances, 10, 2880]
				y = np.concatenate(y, axis=0)   # [total_instances]

			if len(x) == 0:
				continue

			self.instances.append(x)
			self.labels.append(y)

		if not self.instances:
			raise ValueError("No valid instances found for the provided subject IDs.")

		# Stack into one array
		self.instances = np.concatenate(
			self.instances, axis=0).astype(np.float32)
		self.labels = np.concatenate(self.labels,    axis=0).astype(np.float32)

		print(f"Dataset: {len(self.labels)} instances | "
					f"Aggression rate: {self.labels.mean()*100:.1f}%")

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		# [n_channels, seq_len] = [10, 2880]
		x = self.instances[idx]                   

		# transpose to: [seq_len, n_channels] = [2880, 10]
		x = torch.from_numpy(x).T              

		# scalar float for bce
		y = torch.tensor(self.labels[idx])     

		return x, y


def get_dataloaders(bin_file_path, train_ids, val_ids, test_ids,
                    batch_size=32, num_workers=0):
	"""
	Loads .bin file, return train/val/test dataloaders

	Parameters
	bin_file_path : str
	train_ids / val_ids / test_ids : list of str
	batch_size : int
		Number of instances per batch. Default 32.
	num_workers : int

	Returns
	train_loader, val_loader, test_loader : DataLoader
			Each yields (x, y) batches where:
			x shape: [batch_size, 2880, 10]  
			y shape: [batch_size]
	pos_weight : torch.Tensor
	signal_cols : list of str
	"""

	print(f"Loading .bin file: {bin_file_path}")
	with open(bin_file_path, 'rb') as f:
		datalist = pickle.load(f)

	# unpack
	(dict_of_instances, dict_of_labels, id_blacklist,
		superposition_lists, signal_cols, session_dfs) = datalist

	print(f"  Subjects loaded:     {len(dict_of_instances)}")
	print(f"  Blacklisted:         {id_blacklist}")
	print(f"  Signal channels:     {signal_cols}")

	# Remove blacklisted subjects
	def clean(ids):
			return [s for s in ids if s not in id_blacklist]

	train_ids = clean(train_ids)
	val_ids = clean(val_ids)
	test_ids = clean(test_ids)

	print("\nBuilding datasets...")
	print(f"  Train subjects: {len(train_ids)}")
	train_ds = AggDatasetPatchTST(dict_of_instances, dict_of_labels, train_ids)
	print(f"  Val subjects:   {len(val_ids)}")
	val_ds = AggDatasetPatchTST(dict_of_instances, dict_of_labels, val_ids)
	print(f"  Test subjects:  {len(test_ids)}")
	test_ds = AggDatasetPatchTST(dict_of_instances, dict_of_labels, test_ids)

	# Compute positive class weight for imbalanced data
	# pos_weight = n_negative / n_positive
	n_pos = train_ds.labels.sum()
	n_neg = len(train_ds.labels) - n_pos
	pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
	print(f"\n  Class imbalance — pos_weight for loss: {pos_weight.item():.2f}")

	train_loader = DataLoader(train_ds, batch_size=batch_size,
														shuffle=True,  num_workers=num_workers)
	val_loader = DataLoader(val_ds,   batch_size=batch_size,
													shuffle=False, num_workers=num_workers)
	test_loader = DataLoader(test_ds,  batch_size=batch_size,
														shuffle=False, num_workers=num_workers)

	return train_loader, val_loader, test_loader, pos_weight, signal_cols
