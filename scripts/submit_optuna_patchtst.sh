#!/bin/bash
# Submit N independent Optuna worker jobs to SLURM for PatchTST tuning.
# Each job runs 1 trial and reports to a shared SQLite study.
# Pruning kills bad trials after 5 epochs (~30 min instead of ~1 hr).
#
# Usage:
#   bash scripts/submit_optuna_patchtst.sh        # default 8 jobs
#   bash scripts/submit_optuna_patchtst.sh 20     # 20 jobs

N_JOBS=${1:-8}

DATA_PATH="/scratch/borasaniya.t/CBS_DATA_ASD_ONLY"
SAVE_PATH="experiments/results/patchtst/optuna"
LOG_DIR="experiments/logs/patchtst_optuna"
STUDY_DB="$(realpath "$SAVE_PATH")/study.db"

mkdir -p "$SAVE_PATH" "$LOG_DIR"

echo "Submitting $N_JOBS Optuna worker jobs for PatchTST"
echo "  Study DB: $STUDY_DB"
echo ""

for i in $(seq 1 "$N_JOBS"); do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=patchtst_optuna_${i}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=${LOG_DIR}/output_${i}_%j.txt
#SBATCH --error=${LOG_DIR}/error_${i}_%j.txt
#SBATCH --chdir=/home/ma.yun2/ASD_agg_deep_learning
#SBATCH --mail-user=ma_yun2@northeastern.edu
#SBATCH --mail-type=FAIL

export PYTHONUNBUFFERED=1

module load miniconda3/25.9.1
eval "\$(conda shell.bash hook)"
conda activate asd_agg_dl

export PYTHONPATH="\$(realpath shared):\$PYTHONPATH"

python models/patchtst/optuna_search.py \
    --data_path $DATA_PATH \
    --study_path $STUDY_DB \
    --save_path $SAVE_PATH \
    --n_trials 1 \
    --prune_epochs 5 \
    --total_epochs 15 \
    --cuda \
    --gpu 0
EOF
    echo "  Submitted job $i"
done

echo ""
echo "All $N_JOBS jobs submitted."
echo "Monitor: squeue -u ma.yun2"
echo "Check progress:"
echo "  python -c \"import optuna; s=optuna.load_study(study_name='patchtst_aggression', storage='sqlite:///$STUDY_DB'); print(f'Trials: {len(s.trials)}, Best: {s.best_value:.4f}')\""
