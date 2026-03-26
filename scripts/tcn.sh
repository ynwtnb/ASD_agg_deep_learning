#!/bin/bash
#SBATCH -J tcn_training
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o output_tcn_%j.txt
#SBATCH -e error_tcn_%j.txt
#SBATCH --mail-user=borasaniya.t@northeastern.edu
#SBATCH --mail-type=ALL

DATA_PATH="/scratch/borasaniya.t/CBS_DATA_ASD_ONLY/"
SAVE_PATH="../experiments/results/tcn"
HYPER_PATH="../models/tcn/default_parameters.json"
BIN_SIZE=15
N_OBS_FRAMES=12
N_PRED_FRAMES=4
SPLIT="loso"
N_SPLITS=5
VAL_PROP=0.2
GPU=0

module load miniconda3/25.9.1
conda activate asd_agg_dl

python ../models/tcn/pipeline.py \
    --data_path $DATA_PATH \
    --save_path $SAVE_PATH \
    --hyper $HYPER_PATH \
    --bin_size $BIN_SIZE \
    --num_observation_frames $N_OBS_FRAMES \
    --num_prediction_frames $N_PRED_FRAMES \
    --split $SPLIT \
    --n_splits $N_SPLITS \
    --val_prop $VAL_PROP \
    --cuda \
    --gpu $GPU
