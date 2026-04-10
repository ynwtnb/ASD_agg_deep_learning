#!/bin/bash
#SBATCH --job-name=tcn_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --array=0-7
#SBATCH --output=experiments/logs/tcn_sweep_%A_%a.out
#SBATCH --error=experiments/logs/tcn_sweep_%A_%a.err
#SBATCH --chdir=/home/borasaniya.t/ASD_agg_deep_learning

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate asd_agg_dl

CONFIGS=(
    "lr0.0001_do0.1_pw5.0"
    "lr0.0001_do0.2_pw5.0"
    "lr0.0005_do0.1_pw5.0"
    "lr0.0005_do0.2_pw5.0"
    "lr0.001_do0.1_pw5.0"
    "lr0.001_do0.2_pw5.0"
    "lr0.0005_do0.1_pw3.0"
    "lr0.0005_do0.2_pw3.0"
)

CONFIG_NAME=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
CONFIG_PATH="models/tcn/sweep_configs/${CONFIG_NAME}.json"
SAVE_PATH="experiments/results/tcn/sweep/${CONFIG_NAME}"

echo "=== Job ${SLURM_ARRAY_TASK_ID}: ${CONFIG_NAME} ==="
echo "Config: ${CONFIG_PATH}"
echo "Output: ${SAVE_PATH}"

python models/tcn/pipeline.py \
    --data_path /scratch/borasaniya.t/CBS_DATA_ASD_ONLY \
    --save_path "$SAVE_PATH" \
    --hyper "$CONFIG_PATH" \
    --split session \
    --cuda \
    --gpu 0