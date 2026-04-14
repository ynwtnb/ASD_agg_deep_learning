#!/bin/bash
# Submit N independent Optuna worker jobs to SLURM.
# Each job runs one trial and reports to a shared SQLite study.
#
# Usage:
#   bash submit_optuna.sh <n_jobs>
#   bash submit_optuna.sh 20

N_JOBS=${1:-10}

DATA_PATH="/scratch/borasaniya.t/CBS_DATA_ASD_ONLY/"
SAVE_PATH="../experiments/results/shapenet_tuning"
STORAGE="sqlite:////$(realpath ../experiments/results/shapenet_tuning/optuna.db)"
STUDY_NAME="shapenet_tuning"
CLUSTER_NUM=20
SPLIT="session"
BIN_SIZE=15
N_OBS_FRAMES=12
N_PRED_FRAMES=12
MAX_DISCOVERY_SAMPLES=500
VAL_PROP=0.2
STRIDE=6
GPU=0

mkdir -p "$SAVE_PATH"

echo "Submitting $N_JOBS Optuna worker jobs"
echo "  Study:   $STUDY_NAME"
echo "  Storage: $STORAGE"
echo "  Split:   $SPLIT"
echo ""

for i in $(seq 1 "$N_JOBS"); do
    sbatch <<EOF
#!/bin/bash
#SBATCH -J shapenet_optuna_${i}
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH -o output_optuna_${i}_%j.txt
#SBATCH -e error_optuna_${i}_%j.txt
#SBATCH --mail-user=watanabe.y@northeastern.edu
#SBATCH --mail-type=FAIL

module load miniconda3/25.9.1
conda activate asd_agg_dl
export PATH="/home/watanabe.y/.conda/envs/asd_agg_dl/bin:\$PATH"

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
    --cuda --gpu $GPU
EOF
    echo "  Submitted job $i"
done

echo ""
echo "All $N_JOBS jobs submitted."
echo "Monitor progress:"
echo "  optuna-dashboard $STORAGE"
