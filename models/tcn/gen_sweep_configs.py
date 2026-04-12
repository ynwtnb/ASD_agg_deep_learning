"""Generates the 8 sweep configuration JSON files for stride subsampling sweep."""

import json
import os

SWEEP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_configs')
os.makedirs(SWEEP_DIR, exist_ok=True)

# stable configuration from sweep 1:
#   BatchNorm, no balanced sampling (shuffle=True), pos_weight=1.0,
#   dropout=0.2, lr=1e-3, weight_decay=1e-4
#
# problem: severe overfitting due to 91.7% overlap between consecutive
# training instances (~61K instances but far fewer independent examples)
#
# fix: stride subsampling within each session before DataLoader construction
#   stride=1  -> 61K instances (no subsampling, sweep 1 reference)
#   stride=4  -> ~15K instances, ~33% overlap between adjacent kept instances
#   stride=6  -> ~10K instances, minimal overlap
#   stride=8  -> ~7.5K instances, near-zero overlap
#   stride=12 -> ~5K instances, zero overlap (each kept instance is independent)
#
# also test dropout=0.3 with natural batch composition (no balanced sampling)
# to see if higher regularization is now stable

ARCH = [64, 64, 128, 128, 256, 256, 256, 256]

BASE = {
    "n_input_channels": 10,
    "channel_list": ARCH,
    "kernel_size": 7,
    "readout": "mean",
    "batch_size": 32,
    "epochs": 50,
    "patience": 10,
    "max_grad_norm": 1.0,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
}

GRID = [
    # reference: no subsampling, same as sweep 1 winner
    {"train_stride": 1,  "dropout": 0.2, "tag": "stride1_do0.2"},
    # stride subsampling with known-stable dropout
    {"train_stride": 4,  "dropout": 0.2, "tag": "stride4_do0.2"},
    {"train_stride": 6,  "dropout": 0.2, "tag": "stride6_do0.2"},
    {"train_stride": 8,  "dropout": 0.2, "tag": "stride8_do0.2"},
    {"train_stride": 12, "dropout": 0.2, "tag": "stride12_do0.2"},
    # test whether higher dropout is stable without balanced sampling
    {"train_stride": 4,  "dropout": 0.3, "tag": "stride4_do0.3"},
    {"train_stride": 6,  "dropout": 0.3, "tag": "stride6_do0.3"},
    {"train_stride": 8,  "dropout": 0.3, "tag": "stride8_do0.3"},
]

for i, combo in enumerate(GRID):
    tag = combo.pop("tag")
    config = {**BASE, **combo}
    path = os.path.join(SWEEP_DIR, f"{tag}.json")
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"  [{i}] {path}")

print(f"\nGenerated {len(GRID)} configs in {SWEEP_DIR}")