#!/bin/bash
#SBATCH -J data_extraction                            # Job name
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 3                               # Number of tasks
#SBATCH --time=1:00:00                      # Time limit (HH:MM:SS)
#SBATCH --partition=short                   # Partition
#SBATCH -o output_shapenet_%j.txt                    # Standard output file
#SBATCH -e error_shapenet_%j.txt                     # Standard error file
#SBATCH --mail-user=$watanabe.y@northeastern.edu  # Email
#SBATCH --mail-type=ALL                     # Type of email notifications

DATA_PATH="/scratch/borasaniya.t/CBS_DATA_ASD_ONLY/"
SAVE_PATH="../experiments/results/shapenet"
HYPER_PATH="../models/shapenet/default_parameters.json"
BIN_SIZE=15
N_OBS_FRAMES=12
N_PRED_FRAMES=12
CLUSTER_NUM=10
SPLIT="kfold"
N_SPLITS=1
GPU=0

conda activate asd_agg_dl
export PATH="/home/watanabe.y/.conda/envs/asd_agg_dl/bin:$PATH"
python ../models/shapenet/pipeline.py --data_path $DATA_PATH --save_path $SAVE_PATH --hyper $HYPER_PATH --bin_size $BIN_SIZE --num_observation_frames $N_OBS_FRAMES --num_prediction_frames $N_PRED_FRAMES --cluster_num $CLUSTER_NUM --split $SPLIT --n_splits $N_SPLITS --cuda --gpu $GPU
