#!/bin/bash
# Resume interrupted Optuna trials that are stuck in RUNNING state.
#
# For each RUNNING trial:
#   1. Marks it as FAILED in the Optuna DB
#   2. Submits a new sbatch job that resumes from its saved checkpoints
#      (encoder / shapelets / features already on disk are reused automatically)
#
# Usage:
#   bash resume_optuna_shapenet.sh

DATA_PATH="/scratch/borasaniya.t/CBS_DATA_ASD_ONLY/"
SAVE_PATH="../experiments/results/shapenet_tuning"
LOG_DIR="../experiments/logs/shapenet_tuning"
mkdir -p "$SAVE_PATH" "$LOG_DIR"
STORAGE="sqlite:////$(realpath "$SAVE_PATH")/optuna.db"
STUDY_NAME="shapenet_tuning"
CLUSTER_NUM=50
SPLIT="session"
BIN_SIZE=15
N_OBS_FRAMES=12
N_PRED_FRAMES=12
MAX_DISCOVERY_SAMPLES=2000
VAL_PROP=0.2
STRIDE=6
GPU=0

echo "Looking for RUNNING trials in: $STORAGE"
echo ""

# Find RUNNING trial numbers, mark them as FAILED, print their directories.
TRIAL_DIRS=$(/home/watanabe.y/.conda/envs/asd_agg_dl/bin/python3 - <<PYEOF
import sys, os
sys.path.insert(0, '../models/shapenet')
import optuna
from optuna.trial import TrialState

storage   = "$STORAGE"
study_name = "$STUDY_NAME"
save_path  = "$(realpath "$SAVE_PATH")"

import time

RECENT_MINUTES = 60  # treat as still-running if any file modified within this window

# Files written by the resume script itself — exclude from modification check
EXCLUDE = {'trial_params.json', 'trial_run_config.json'}

def recently_modified(trial_dir, minutes=RECENT_MINUTES):
    """True if any training file (not bookkeeping files) was modified recently."""
    cutoff = time.time() - minutes * 60
    try:
        for fname in os.listdir(trial_dir):
            if fname in EXCLUDE:
                continue
            fpath = os.path.join(trial_dir, fname)
            if os.path.isfile(fpath) and os.path.getmtime(fpath) > cutoff:
                return True
    except FileNotFoundError:
        pass
    return False

study = optuna.load_study(study_name=study_name, storage=storage)

for t in study.trials:
    if t.state != TrialState.RUNNING:
        continue
    trial_dir   = f"{save_path}/trial_{t.number}"
    params_file = f"{trial_dir}/trial_params.json"
    if not os.path.exists(params_file):
        # Params not saved yet (job interrupted early) — reconstruct from Optuna DB
        if not t.params:
            print(f"[skip] trial {t.number}: no params in DB either", file=sys.stderr)
            continue
        os.makedirs(trial_dir, exist_ok=True)
        import json as _json
        db_params = dict(t.params)
        db_params.setdefault('compared_length', None)
        with open(params_file, 'w') as f:
            _json.dump({'trial_number': t.number, 'params': db_params}, f, indent=2)
        print(f"[restored] trial {t.number}: trial_params.json written from DB", file=sys.stderr)
    if recently_modified(trial_dir):
        print(f"[skip] trial {t.number}: files modified within last {RECENT_MINUTES} min, likely still running", file=sys.stderr)
        continue
    study._storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
    print(f"[mark FAILED] trial {t.number}", file=sys.stderr)
    print(trial_dir)
PYEOF
)

if [ -z "$TRIAL_DIRS" ]; then
    echo "No resumable RUNNING trials found."
    exit 0
fi

N=0
while IFS= read -r trial_dir; do
    [ -z "$trial_dir" ] && continue
    N=$((N + 1))
    TRIAL_NAME=$(basename "$trial_dir")
    LOG_ABS=$(realpath "$LOG_DIR")
    TRIAL_DIR_ABS=$(realpath "$trial_dir")
    echo "  Submitting resume job for: $trial_dir"

    sbatch <<EOF
#!/bin/bash
#SBATCH -J shapenet_resume_${TRIAL_NAME}
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --mem=128G
#SBATCH -o ${LOG_ABS}/output_resume_${TRIAL_NAME}_%j.txt
#SBATCH -e ${LOG_ABS}/error_resume_${TRIAL_NAME}_%j.txt
#SBATCH --mail-user=watanabe.y@northeastern.edu
#SBATCH --mail-type=FAIL

module load miniconda3/25.9.1
eval "\$(conda shell.bash hook)"
conda activate asd_agg_dl
export PATH="/home/watanabe.y/.conda/envs/asd_agg_dl/bin:\$PATH"
export CONDA_PREFIX="/home/watanabe.y/.conda/envs/asd_agg_dl"

python -u ../models/shapenet/optuna_worker.py \\
    --data_path $DATA_PATH \\
    --save_path $SAVE_PATH \\
    --storage "$STORAGE" \\
    --study_name $STUDY_NAME \\
    --n_trials 1 \\
    --split $SPLIT \\
    --cluster_num $CLUSTER_NUM \\
    --val_prop $VAL_PROP \\
    --bin_size $BIN_SIZE \\
    --num_observation_frames $N_OBS_FRAMES \\
    --num_prediction_frames $N_PRED_FRAMES \\
    --max_discovery_samples $MAX_DISCOVERY_SAMPLES \\
    --stride $STRIDE \\
    --resume_trial_dir ${TRIAL_DIR_ABS} \\
    --cuda --gpu $GPU
EOF
done <<< "$TRIAL_DIRS"

echo ""
echo "Submitted $N resume jobs."
