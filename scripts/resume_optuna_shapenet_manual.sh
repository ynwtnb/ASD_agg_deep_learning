#!/bin/bash
# Manually rerun specific Optuna trials by trial number.
#
# Usage:
#   bash rerun_trials_shapenet.sh 1 5 12
# (specify any number of trial numbers as arguments)

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

if [ $# -eq 0 ]; then
    echo "Usage: bash $(basename $0) <trial_num> [trial_num ...]"
    echo "Example: bash $(basename $0) 1 5 12"
    exit 1
fi

N=0
for TRIAL_NUM in "$@"; do
    TRIAL_DIR="$(realpath "$SAVE_PATH")/trial_${TRIAL_NUM}"

    if [ ! -f "${TRIAL_DIR}/trial_params.json" ]; then
        echo "[skip] trial ${TRIAL_NUM}: ${TRIAL_DIR}/trial_params.json not found"
        continue
    fi

    TRIAL_NAME="trial_${TRIAL_NUM}"
    LOG_ABS=$(realpath "$LOG_DIR")
    echo "  Submitting rerun job for: trial ${TRIAL_NUM} (dir: ${TRIAL_DIR})"

    sbatch <<EOF
#!/bin/bash
#SBATCH -J shapenet_rerun_${TRIAL_NAME}
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --mem=128G
#SBATCH -o ${LOG_ABS}/output_rerun_${TRIAL_NAME}_%j.txt
#SBATCH -e ${LOG_ABS}/error_rerun_${TRIAL_NAME}_%j.txt
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
    --resume_trial_dir ${TRIAL_DIR} \\
    --cuda --gpu $GPU
EOF

    N=$((N + 1))
done

echo ""
echo "Submitted $N rerun jobs."