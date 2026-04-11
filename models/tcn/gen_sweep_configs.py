"""Generates the 8 sweep configuration JSON files for capacity/regularization sweep."""

import json
import os

SWEEP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_configs')
os.makedirs(SWEEP_DIR, exist_ok=True)

# lr=1e-3 fixed from round 1 results (best AUPRC and AUROC)
# kernel_size=7 with 8 blocks gives RF=3061, covering full 2880-sample window
# two architectures to test capacity reduction:
#   A = [32,32,64,64,128,128,128,128]  ~1M params (narrow pyramid)
#   B = [32,32,32,32,64,64,64,64]      ~0.3M params (very narrow)

ARCH_A = [32, 32, 64, 64, 128, 128, 128, 128]
ARCH_B = [32, 32, 32, 32, 64, 64, 64, 64]

BASE = {
    "n_input_channels": 10,
    "kernel_size": 7,
    "readout": "mean",
    "batch_size": 32,
    "epochs": 50,
    "patience": 10,
    "max_grad_norm": 1.0,
    "lr": 1e-3,
    "warmup_epochs": 5,
}

GRID = [
    # arch A: narrow pyramid, dropout x weight_decay
    {"channel_list": ARCH_A, "dropout": 0.3, "weight_decay": 1e-3, "tag": "archA_do0.3_wd1e-3"},
    {"channel_list": ARCH_A, "dropout": 0.3, "weight_decay": 1e-2, "tag": "archA_do0.3_wd1e-2"},
    {"channel_list": ARCH_A, "dropout": 0.4, "weight_decay": 1e-3, "tag": "archA_do0.4_wd1e-3"},
    {"channel_list": ARCH_A, "dropout": 0.4, "weight_decay": 1e-2, "tag": "archA_do0.4_wd1e-2"},
    # arch B: very narrow, dropout x weight_decay
    {"channel_list": ARCH_B, "dropout": 0.3, "weight_decay": 1e-3, "tag": "archB_do0.3_wd1e-3"},
    {"channel_list": ARCH_B, "dropout": 0.3, "weight_decay": 1e-2, "tag": "archB_do0.3_wd1e-2"},
    {"channel_list": ARCH_B, "dropout": 0.4, "weight_decay": 1e-3, "tag": "archB_do0.4_wd1e-3"},
    {"channel_list": ARCH_B, "dropout": 0.4, "weight_decay": 1e-2, "tag": "archB_do0.4_wd1e-2"},
]

for i, combo in enumerate(GRID):
    tag = combo.pop("tag")
    config = {**BASE, **combo}
    path = os.path.join(SWEEP_DIR, f"{tag}.json")
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"  [{i}] {path}")

print(f"\nGenerated {len(GRID)} configs in {SWEEP_DIR}")