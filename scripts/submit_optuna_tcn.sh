#!/bin/bash
# Submit N independent Optuna worker jobs to SLURM for TCN tuning.
# Each job runs 1 trial and reports to a shared SQLite study.
# Pruning kills bad trials after 5 epochs (~30 min instead of 2-3 hrs).
#
# Usage:
#   bash scripts/submit_optuna_tcn.sh        # default 12 jobs
#   bash scripts/submit_optuna_tcn.sh 20     # 20 jobs

N_JOBS=${1:-8}

DATA_PATH="/scratch/borasaniya.t/CBS_DATA_ASD_ONLY"
SAVE_PATH="experiments/results/tcn/optuna"
LOG_DIR="experiments/logs/tcn_optuna"
STUDY_DB="$(realpath "$SAVE_PATH")/study.db"

mkdir -p "$SAVE_PATH" "$LOG_DIR"

echo "Submitting $N_JOBS Optuna worker jobs for TCN"
echo "  Study DB: $STUDY_DB"
echo ""

for i in $(seq 1 "$N_JOBS"); do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tcn_optuna_${i}
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
    --n_trials 1 \
    --cuda \
    --gpu 0
EOF
    echo "  Submitted job $i"
done

echo ""
echo "All $N_JOBS jobs submitted."
echo "Monitor: squeue -u borasaniya.t"
echo "Check progress:"
echo "  python -c \"import optuna; s=optuna.load_study(study_name='tcn_aggression', storage='sqlite:///$STUDY_DB'); print(f'Trials: {len(s.trials)}, Best: {s.best_value:.4f}')\""
