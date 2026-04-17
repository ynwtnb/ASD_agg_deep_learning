#!/bin/bash
#SBATCH -J patchtst                                   # Job name
#SBATCH -N 1                                          # Number of nodes
#SBATCH -n 4                                          # Number of tasks
#SBATCH --time=8:00:00                                # Time limit (HH:MM:SS)
#SBATCH --partition=gpu                               # GPU partition 
#SBATCH --gres=gpu:1                                  # Request 1 GPU
#SBATCH --mem=32G                                     # Memory
#SBATCH -o output_patchtst_%j.txt                     # Standard output file
#SBATCH -e error_patchtst_%j.txt                      # Standard error file
#SBATCH --mail-user=ma.yun2@northeastern.edu          # Email
#SBATCH --mail-type=ALL                               # Email on start, end, fail

DATA_PATH="/scratch/borasaniya.t/CBS_DATA_ASD_ONLY/"
SAVE_PATH="${SAVE_PATH:-../experiments/results/patchtst}"
HYPER_PATH="${HYPER_PATH:-../models/patchtst/default_parameters.json}"
BIN_SIZE=15
N_OBS_FRAMES=12
N_PRED_FRAMES=12
SPLIT="${SPLIT:-session}"
N_SPLITS=3
GPU=0

module load miniconda3/25.9.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate asd_agg_dl
export PYTHONPATH="$(realpath ../shared):$PYTHONPATH"

python -u ../models/patchtst/pipeline.py \
    --data_path $DATA_PATH \
    --save_path $SAVE_PATH \
    --hyper $HYPER_PATH \
    --bin_size $BIN_SIZE \
    --num_observation_frames $N_OBS_FRAMES \
    --num_prediction_frames $N_PRED_FRAMES \
    --split $SPLIT \
    --n_splits $N_SPLITS \
    --cuda \
    --gpu $GPU