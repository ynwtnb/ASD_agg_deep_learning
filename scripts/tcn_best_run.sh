#!/bin/bash
# Run pipeline.py with the best Optuna config for a given window size.
#
# Usage:
#   bash scripts/tcn_best_run.sh 12 12   # 3-min window
#   bash scripts/tcn_best_run.sh 8  8    # 2-min window
#   bash scripts/tcn_best_run.sh 4  4    # 1-min window

N_OBS_FRAMES=${1:-12}
N_PRED_FRAMES=${2:-$N_OBS_FRAMES}
TRIAL_NUMBER=${3:-0}
SEED=$((42 + TRIAL_NUMBER))

WINDOW_TAG="obs${N_OBS_FRAMES}_pred${N_PRED_FRAMES}"
DATA_PATH="/scratch/borasaniya.t/CBS_DATA_ASD_ONLY"
HYPER="experiments/results/tcn/optuna/${WINDOW_TAG}/best_config.json"
SAVE_PATH="/scratch/borasaniya.t/CBS_DATA_ASD_ONLY/runs/${WINDOW_TAG}/best_run"
LOG_DIR="experiments/logs"

mkdir -p "$SAVE_PATH" "$LOG_DIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tcn_best_${WINDOW_TAG}
#SBATCH --chdir=/home/borasaniya.t/ASD_agg_deep_learning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=${LOG_DIR}/tcn_best_${WINDOW_TAG}_%j.out
#SBATCH --error=${LOG_DIR}/tcn_best_${WINDOW_TAG}_%j.err
#SBATCH --mail-user=borasaniya.t@northeastern.edu
#SBATCH --mail-type=END,FAIL

export PYTHONUNBUFFERED=1

module load miniconda3/25.9.1
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate asd_agg_dl

python models/tcn/pipeline.py \
    --data_path "$DATA_PATH" \
    --save_path "$SAVE_PATH" \
    --hyper "$HYPER" \
    --num_observation_frames $N_OBS_FRAMES \
    --num_prediction_frames $N_PRED_FRAMES \
    --seed $SEED \
    --split session \
    --cuda \
    --gpu 0
EOF

echo "submitted best_run job for ${WINDOW_TAG}"
echo "  hyper:     $HYPER"
echo "  save_path: $SAVE_PATH"