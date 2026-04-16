#!/bin/bash
# Submit N independent Optuna worker jobs to SLURM for TCN tuning.
#
# Usage:
#   bash scripts/submit_optuna_tcn.sh           # 3-min window, 12 jobs
#   bash scripts/submit_optuna_tcn.sh 8 8 12    # 2-min window, 12 jobs
#   bash scripts/submit_optuna_tcn.sh 4 4 8     # 1-min window, 8 jobs

N_OBS_FRAMES=${1:-12}
N_PRED_FRAMES=${2:-12}
N_JOBS=${3:-12}

DATA_PATH="/scratch/borasaniya.t/CBS_DATA_ASD_ONLY"
WINDOW_TAG="obs${N_OBS_FRAMES}_pred${N_PRED_FRAMES}"
SAVE_PATH="experiments/results/tcn/optuna/${WINDOW_TAG}"
LOG_DIR="experiments/logs/tcn_optuna_${WINDOW_TAG}"
STUDY_DB="$(realpath --canonicalize-missing "$SAVE_PATH")/study.db"

mkdir -p "$SAVE_PATH" "$LOG_DIR"

echo "Submitting $N_JOBS Optuna worker jobs | window=${WINDOW_TAG}"
echo "  Study DB: $STUDY_DB"
echo ""

for i in $(seq 1 "$N_JOBS"); do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tcn_optuna_${WINDOW_TAG}_${i}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=${LOG_DIR}/output_${i}_%j.txt
#SBATCH --error=${LOG_DIR}/error_${i}_%j.txt
#SBATCH --chdir=/home/borasaniya.t/ASD_agg_deep_learning

export PYTHONUNBUFFERED=1

module load miniconda3/25.9.1
eval "\$(conda shell.bash hook)"
conda activate asd_agg_dl

python models/tcn/optuna_search.py \
    --data_path $DATA_PATH \
    --study_path $STUDY_DB \
    --save_path $SAVE_PATH \
    --num_observation_frames $N_OBS_FRAMES \
    --num_prediction_frames $N_PRED_FRAMES \
    --n_trials 1 \
    --cuda \
    --gpu 0
EOF
    echo "  submitted job $i"
done

echo ""
echo "all $N_JOBS jobs submitted."
echo "monitor: squeue -u borasaniya.t"