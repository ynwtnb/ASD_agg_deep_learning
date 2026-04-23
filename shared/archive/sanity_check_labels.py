"""
Sanity check for label assignment in the ShapeNet pipeline.

Loads the saved bin cache (bin_feat_<bin_size>S.b) and traces labels through
each stage to identify where positive labels disappear.

Usage:
    python sanity_check_labels.py --data_path <path> --bin_size 15 \
        --num_observation_frames 12 --num_prediction_frames 12
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../shared'))
from data_extraction import gen_instances_from_raw_feat_dictionary


def check_bin_cache(data_path, bin_size):
    """Stage 1: check raw bin cache for positive labels."""
    cache_file = os.path.join(data_path, f'bin_feat_{bin_size}S.b')
    if not os.path.isfile(cache_file):
        print(f"[ERROR] Cache file not found: {cache_file}")
        print("        Run with --run_from_scratch first.")
        return None

    print(f"\n=== Stage 1: Raw bin cache ({cache_file}) ===")
    with open(cache_file, 'rb') as f:
        data_dict = pickle.load(f)

    total_bins = 0
    positive_bins = 0
    participants_with_positives = 0

    for uid, user_data in data_dict.items():
        uid_pos = 0
        for session_labels in user_data['labels']:
            vals = session_labels.values if hasattr(session_labels, 'values') else np.array(session_labels)
            vals = vals.astype(float)
            total_bins += len(vals)
            uid_pos += int((vals > 0).sum())
        positive_bins += uid_pos
        if uid_pos > 0:
            participants_with_positives += 1

    print(f"  Participants:              {len(data_dict)}")
    print(f"  Participants with pos bins:{participants_with_positives} / {len(data_dict)}")
    print(f"  Total bins:                {total_bins}")
    print(f"  Positive bins:             {positive_bins} ({100*positive_bins/max(total_bins,1):.2f}%)")

    if positive_bins == 0:
        print("\n  [!] No positive labels in bin cache.")
        print("      Likely cause: 'Condition' column values do not match agg_cat=['AGG','SIB','ED'].")
        print("      Check a sample CSV file for the actual label values in the 'Condition' column.")
        # Show a sample of Condition values from the first session
        first_uid = next(iter(data_dict))
        print(f"\n  Checking uid={first_uid} label values:")
        for i, lbl in enumerate(data_dict[first_uid]['labels'][:2]):
            vals = lbl.values if hasattr(lbl, 'values') else np.array(lbl)
            print(f"    session {i}: unique values = {np.unique(vals.astype(float))}, "
                  f"shape = {vals.shape}")

    return data_dict


def check_instance_generation(data_dict, n_obs, n_pred, bin_size):
    """Stage 2: check labels after gen_instances_from_raw_feat_dictionary."""
    print(f"\n=== Stage 2: Instance generation (obs={n_obs}, pred={n_pred}) ===")

    # Debug: check index alignment between bin_df and bin_labels for first subject/session
    print("\n  [index alignment check]")
    first_uid = next(iter(data_dict))
    first_features = data_dict[first_uid]['features'][0]
    first_labels   = data_dict[first_uid]['labels'][0]
    print(f"    bin_df index type:     {type(first_features.index)}")
    print(f"    bin_df index names:    {getattr(first_features.index, 'names', 'n/a')}")
    print(f"    bin_df index sample:   {first_features.index[:3].tolist()}")
    print(f"    bin_labels index type: {type(first_labels.index)}")
    print(f"    bin_labels index sample: {first_labels.index[:3].tolist()}")
    # Use Timestamp level for alignment (same logic as the fix in data_extraction.py)
    if isinstance(first_features.index, pd.MultiIndex):
        feat_ts = first_features.index.get_level_values('Timestamp')
    else:
        feat_ts = first_features.index
    reindexed = first_labels.reindex(feat_ts).fillna(0)
    print(f"    bin_labels unique (raw):       {first_labels.unique()[:10]}")
    print(f"    bin_labels unique (reindexed): {reindexed.unique()[:10]}")
    overlap = feat_ts.isin(first_labels.index).sum()
    print(f"    overlapping timestamps: {overlap} / {len(first_features)}")

    result = gen_instances_from_raw_feat_dictionary(
        data_dict, n_obs, n_pred,
        o_multiclass=False,
        o_return_list_of_sessions=False,
        outdir='/tmp',           # avoid overwriting real cache
        o_run_from_scratch=True,
        bin_size=bin_size,
        print_progress=False,
    )
    dict_of_instances, dict_of_labels = result[0], result[1]

    total = 0
    positive = 0
    for uid, lbl in dict_of_labels.items():
        lbl = np.array(lbl)
        if lbl.ndim > 1:
            lbl = lbl[:, 0]
        total += len(lbl)
        positive += int((lbl > 0).sum())

    print(f"  Total instances:   {total}")
    print(f"  Positive instances:{positive} ({100*positive/max(total,1):.2f}%)")
    print(f"  Negative instances:{total - positive}")

    if positive == 0:
        print("\n  [!] No positive instances after gen_instances_from_raw_feat_dictionary.")
        print("      Even though bin labels may exist, the prediction window (future bins)")
        print("      contains no aggression — or all windows with ongoing aggression are skipped.")
        print(f"      Check: are there aggression events that last longer than {n_pred} bins")
        print(f"      ({n_pred * bin_size}s = {n_pred * bin_size / 60:.1f} min) after a calm period?")

        # Show per-participant label distribution
        print("\n  Per-participant instance counts:")
        for uid, lbl in list(dict_of_labels.items())[:5]:
            lbl = np.array(lbl)
            if lbl.ndim > 1:
                lbl = lbl[:, 0]
            print(f"    uid={uid}: {len(lbl)} instances, {int((lbl>0).sum())} positive")

    return dict_of_labels


def main():
    parser = argparse.ArgumentParser(description='Sanity check label assignment')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--bin_size', type=int, default=15)
    parser.add_argument('--num_observation_frames', type=int, default=12)
    parser.add_argument('--num_prediction_frames', type=int, default=12)
    args = parser.parse_args()

    data_dict = check_bin_cache(args.data_path, args.bin_size)
    if data_dict is None:
        return

    check_instance_generation(
        data_dict,
        args.num_observation_frames,
        args.num_prediction_frames,
        args.bin_size,
    )


if __name__ == '__main__':
    main()
