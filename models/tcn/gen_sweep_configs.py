"""Generates the 8 sweep configuration JSON files for pos_weight + BatchNorm sweep."""

import json
import os

SWEEP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_configs')
os.makedirs(SWEEP_DIR, exist_ok=True)

# previous sweeps established:
#   - BatchNorm + balanced sampling is unstable at dropout > 0.2
#   - BatchNorm + shuffle=True + pos_weight should be stable because
#     natural batch composition gives BatchNorm consistent statistics
#   - lr=1e-3 is the right regime, lr=5e-4 also works
#   - raw pos_weight is ~4.0 for this dataset
#
# this sweep: shuffle=True, clamped pos_weight, push dropout and weight_decay
# to control the severe overfitting observed in sweep 1

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
    "warmup_epochs": 5,
    "max_pos_weight": 10.0,
}

GRID = [
    # baseline reference: sweep 1 winner equivalent with pos_weight
    {"dropout": 0.2, "weight_decay": 1e-4, "tag": "do0.2_wd1e-4"},
    # push dropout with light wd
    {"dropout": 0.3, "weight_decay": 1e-4, "tag": "do0.3_wd1e-4"},
    {"dropout": 0.4, "weight_decay": 1e-4, "tag": "do0.4_wd1e-4"},
    {"dropout": 0.5, "weight_decay": 1e-4, "tag": "do0.5_wd1e-4"},
    # push weight_decay with moderate dropout
    {"dropout": 0.3, "weight_decay": 1e-3, "tag": "do0.3_wd1e-3"},
    {"dropout": 0.3, "weight_decay": 5e-3, "tag": "do0.3_wd5e-3"},
    # combined strong regularization
    {"dropout": 0.4, "weight_decay": 1e-3, "tag": "do0.4_wd1e-3"},
    {"dropout": 0.5, "weight_decay": 1e-3, "tag": "do0.5_wd1e-3"},
]

for i, combo in enumerate(GRID):
    tag = combo.pop("tag")
    config = {**BASE, **combo}
    path = os.path.join(SWEEP_DIR, f"{tag}.json")
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"  [{i}] {path}")

print(f"\nGenerated {len(GRID)} configs in {SWEEP_DIR}")