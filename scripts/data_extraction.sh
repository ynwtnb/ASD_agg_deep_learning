#!/bin/bash
#SBATCH -J data_extraction                            # Job name
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                                # Number of tasks
#SBATCH --time=2:00:00                      # Time limit (HH:MM:SS)
#SBATCH --partition=short                   # Partition
#SBATCH -o output_%j.txt                    # Standard output file
#SBATCH -e error_%j.txt                     # Standard error file
#SBATCH --mail-user=$watanabe.y@northeastern.edu  # Email
#SBATCH --mail-type=ALL                     # Type of email notifications

DATA_PATH="/scratch/borasaniya.t/CBS_DATA_ASD_ONLY/"
BIN_SIZE=15
N_OBS_FRAMES=12
N_PRED_FRAMES=12

# Your program/command here
module load miniconda3/25.9.1
conda activate asd_agg_dl
python ../shared/data_extraction.py -dp $DATA_PATH -bs $BIN_SIZE -ofr -no $N_OBS_FRAMES -np $N_PRED_FRAMES
