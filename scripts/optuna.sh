#!/bin/bash
#SBATCH -J optuna                                     # Job name
#SBATCH -N 1                                          # Number of nodes
#SBATCH -n 4                                          # Number of tasks
#SBATCH --time=8:00:00                                # Time limit (HH:MM:SS)
#SBATCH --partition=gpu                               # GPU partition
#SBATCH --gres=gpu:1                                  # Request 1 GPU
#SBATCH --mem=32G                                     # Memory
#SBATCH -o output_optuna_%j.txt                       # Standard output file
#SBATCH -e error_optuna_%j.txt                        # Standard error file
#SBATCH --mail-user=ma.yun2@northeastern.edu          # Email
#SBATCH --mail-type=ALL                               # Email on start, end, fail

DATA_PATH="/scratch/borasaniya.t/CBS_DATA_ASD_ONLY/"
SAVE_PATH="../experiments/results/optuna"
N_TRIALS=5
GPU=0

module load miniconda3/25.9.1
source /shared/EL9/explorer/miniconda3/25.9.1/miniconda3/etc/profile.d/conda.sh
conda activate asd_agg_dl

export PYTHONPATH="$(realpath ../shared):$PYTHONPATH"

pip install optuna -q

python -u ../models/patchtst/optuna_search.py \
    --data_path $DATA_PATH \
    --save_path $SAVE_PATH \
    --n_trials $N_TRIALS \
    --cuda \
    --gpu $GPU
