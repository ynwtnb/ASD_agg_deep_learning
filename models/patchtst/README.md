# PatchTST

A PatchTST-based binary classifier for predicting aggression episodes in ASD patients.

## Model Overview

The model operates in **channel-independent (CI)** mode using the HuggingFace `PatchTSTForClassification` backbone. Each of the 10 physiological channels is treated as an independent token stream. Patches are sliced from each channel, encoded by a shared Transformer encoder, and classified via a **CLS token** whose representations across all channels are concatenated before the linear head.

**Input channels:** BVP, EDA, ACC_X, ACC_Y, ACC_Z, Magnitude, HR, RMSSD, PHASIC, TONIC (10 channels, 16 Hz)

**Input shape:** `[batch, seq_len, 10]` — seq_len = N_OBS_FRAMES × 15s × 16 Hz

**Output:** binary logit (aggression / no aggression)

## Key Hyperparameters (best found via Optuna)

| Parameter | Value |
|---|---|
| d_model | 64 |
| n_layers | 3 |
| n_heads | 8 |
| ffn_dim | 512 |
| patch_len | 32 |
| patch_stride | 16 |
| dropout | 0.476 |
| head_dropout | 0.211 |
| lr | 8.66e-4 |
| pos_weight | 5.32 |

## Window Sizes

| Config file | Window | N_OBS_FRAMES | N_PRED_FRAMES | seq_len |
|---|---|---|---|---|
| `param_optuna_best.json` | 3 min | 12 | 12 | 2880 |
| `param_2min.json` | 2 min | 8 | 8 | 1920 |
| `param_1min.json` | 1 min | 4 | 4 | 960 |

## Training

Run from the `scripts/` directory:

```bash
# Session split (default, 3-min window)
bash patchtst.sh

# Custom window / split
SPLIT=session N_OBS_FRAMES=8 N_PRED_FRAMES=8 \
HYPER_PATH=../models/patchtst/param_2min.json \
SAVE_PATH=../experiments/results/patchtst/session_2min \
bash patchtst.sh
```

**Early stopping** monitors `0.5 × AUROC + 0.5 × AUPRC` (patience=5). Training history and loss curves are saved alongside the model checkpoint.

## Hyperparameter Search (Optuna)

```bash
# Submit 8 parallel Optuna workers
bash submit_optuna_patchtst.sh 8 patchtst_combined
```

Results are stored in `experiments/results/patchtst/optuna_<study_name>/`. Best parameters are saved to `param_optuna_best.json` after manual review.

## Outputs

Each training run saves to `SAVE_PATH/`:

| File | Description |
|---|---|
| `*_model` | Model checkpoint (best epoch by combined score) |
| `*_model_parameters.json` | Full hyperparameter dict |
| `*_history.json` | Per-epoch train loss, val loss, AUROC, AUPRC, score |
| `*_curves.png` | Training curves (loss / AUROC / AUPRC) |

## Interpretability

See `notebooks/patchtst_interpretability.ipynb` for attention weight analysis:
- CLS token attention heatmaps (channel × patch)
- Channel importance ranking
- Temporal attention profile (which part of the window matters)
- Per-layer attention evolution
- Aggression vs. no-aggression difference map
