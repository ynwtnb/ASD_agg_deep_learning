"""Generates the 8 sweep configuration JSON files."""

import json
import os

SWEEP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_configs')
os.makedirs(SWEEP_DIR, exist_ok=True)

BASE = {
    "n_input_channels": 10,
    "channel_list": [64, 64, 128, 128, 256, 256, 256, 256],
    "kernel_size": 7,
    "readout": "mean",
    "batch_size": 32,
    "epochs": 50,
    "patience": 10,
    "max_grad_norm": 1.0,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
}

GRID = [
    {"lr": 1e-4, "dropout": 0.1},
    {"lr": 1e-4, "dropout": 0.2},
    {"lr": 5e-4, "dropout": 0.1},
    {"lr": 5e-4, "dropout": 0.2},
    {"lr": 1e-3, "dropout": 0.1},
    {"lr": 1e-3, "dropout": 0.2},
    {"lr": 5e-4, "dropout": 0.15},
    {"lr": 1e-4, "dropout": 0.15},
]

for i, combo in enumerate(GRID):
    config = {**BASE, **combo}
    name = f"lr{combo['lr']}_do{combo['dropout']}"
    path = os.path.join(SWEEP_DIR, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"  [{i}] {path}")

print(f"\nGenerated {len(GRID)} configs in {SWEEP_DIR}")